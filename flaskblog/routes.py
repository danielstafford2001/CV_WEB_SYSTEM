#some parts based on [1] in bibliography

import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort
from flaskblog import app, db, bcrypt, mail
from flaskblog.forms import (RegistrationForm, LoginForm, UpdateAccountForm,
                             PostForm, RequestResetForm, ResetPasswordForm,NoteForm)
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
import re
from werkzeug.utils import secure_filename
import textract
import glob
import nltk
from collections import defaultdict
from flask_dropzone import Dropzone
import sqlite3
from flaskblog.serve import get_model_api

model_api = get_model_api()


@app.route("/titles")
def titles():
    posts = Post.query.order_by(Post.date_posted.desc())
    titles = []
    for post in posts:
        is_new = True
        for title in titles:
            if title['title'] == post.title:
                title['num'] += 1
                is_new = False
        if is_new:
            title = {}
            title['title'] = post.title
            title['num'] = 1
            titles.append(title)
    return render_template('title.html', titles=titles)

@app.route("/posts/title", methods=['GET', 'POST'])
def posts_by_title():
    if request.method == 'POST':
        post_title = request.form.get('title')
        posts = Post.query.filter_by(title=post_title).all()
    return render_template('posts_title.html', posts=posts)

@app.route("/titles")
def all_titles():
    conn = sqlite3.connect('food.db')
    conn.row_factory = lambda cursor, row: row[0]
    c = conn.cursor()
    ads = c.execute("SELECT title FROM post").fetchall()
    for row in ads:
        print(row)

    return render_template("my_ads.html", ads=ads)

@app.route("/")
@app.route("/home")
def home():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)#pagination- 5 posts per page in date descending order
    return render_template('home.html', posts=posts)

@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')#hashing the password to be stored in the db
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)#a cookie is created to allow remembering of users for a given period of time
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

#this function is used to make the profile pricture fit nicely inot the cirles instead of just placing them in.
def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)

#show a specific post on its own, can only open the ones created by the current user
@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)

@app.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post',
                           form=form, legend='Update Post')

@app.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('home'))


#show all posts by a given user whilest using pagination in descending posting order
@app.route("/user/<string:username>")
def user_posts(username):
    page = request.args.get('page', 1, type=int)
    user = User.query.filter_by(username=username).first_or_404()
    posts = Post.query.filter_by(author=user)\
        .order_by(Post.date_posted.desc())\
        .paginate(page=page, per_page=5)
    return render_template('user_posts.html', posts=posts, user=user)

#trying to send email for password reset at login
def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='danielstafford995@gmail.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)

#trying to send email for password reset at login
@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)

#trying to send email for password reset at login
@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)

#adding notes for a given post
@app.route("/post/<int:post_id>/notes", methods=['GET', 'POST'])
def writing_update_note(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = NoteForm()
    if form.validate_on_submit():
        post.notes= form.notes.data
        db.session.commit()
        flash('Your post notes have been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.notes.data= post.notes
    return render_template('create_note.html', title='Update Notes',
                           form=form, legend='Update Notes')

def nltk_extraction(text):
    entities = defaultdict(list)
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                entities[chunk.label()].append(''.join(c[0] for c in chunk))
    
    my_list = []
    for key, value in entities.items():
        list1= [key]
        my_list.append(list1 + value)
    return "\n\n".join([x[0] + '->' + ', '.join(x[1:]) for x in my_list])


#route that takes file as input and cleans the data (txt,docx,doc,pdf) are accepted and then passes the data onto the new_postfile route which is just a form for a new post that will be pre-populated with the data.
@app.route('/uploader',methods = ['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)
        path= app.config['UPLOAD_FOLDER'] + filename
        pageDOCX = textract.process(path)
        pageDOCX=str(pageDOCX,'utf-8')
        pageDOCX= pageDOCX.replace('\n',' ').strip()
        #file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        content = pageDOCX
        return redirect(url_for('new_postfile'))
        #print(path)

    return render_template('upload.html')

#/Users/danielstafford/Desktop/3rd year/Project/CV Work/CV_WEB_SYSTEM/flaskblog/static
@app.route("/post/newfile", methods=['GET', 'POST'])
@login_required
def new_postfile():
    form = PostForm()
    if form.validate_on_submit():
        pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
        matches = pattern.findall(form.content.data)
        pattern1 = re.compile(
            r'\d{3}[-\.\s]??\d{4}[-\.\s]??\d{4}|\d{5}[-\.\s]??\d{3}[-\.\s]??\d{3}|(?:\d{4}\)?[\s-]?\d{3}[\s-]?\d{4})')
        matches1 = pattern1.findall(form.content.data)

        nltk_result= nltk_extraction(form.content.data)
        post = Post(title=form.title.data, content=form.content.data, author=current_user,email=str(matches), number=str(matches1),entities= nltk_result)
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('home'))
    elif request.method == 'GET':
        folder_path= r'/Users/danielstafford/Desktop/3rd year/Project/CV Work/CV_WEB_SYSTEM/flaskblog/static/*'
        files = glob.glob(folder_path) 
        max_file = max(files, key=os.path.getctime)
        pageDOCX = textract.process(max_file)
        pageDOCX=str(pageDOCX,'utf-8')
        pageDOCX= pageDOCX.replace('\n',' ').strip()

        form.content.data = pageDOCX
    return render_template('create_post.html', title='New Post',
                           form=form, legend='New Post')

#new post via text route
@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
        matches = pattern.findall(form.content.data)
        pattern1 = re.compile(
            r'\d{3}[-\.\s]??\d{4}[-\.\s]??\d{4}|\d{5}[-\.\s]??\d{3}[-\.\s]??\d{3}|(?:\d{4}\)?[\s-]?\d{3}[\s-]?\d{4})')
        matches1 = pattern1.findall(form.content.data)

        #model call here for getting entities back
        nltk_result= nltk_extraction(form.content.data)

        # using model_api to get model predictions
        res = model_api(form.content.data)
        

        post = Post(title=form.title.data, content=form.content.data, author=current_user,email=str(matches), number=str(matches1), 
        entities= nltk_result, entity=str(res))
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('create_post.html', title='New Post',
                           form=form, legend='New Post')
        
#route that takes file as input from drag and drop and cleans the data (txt,docx,doc,pdf) are accepted and then passes the data onto the new_postfile route which is just a form for a new post that will be pre-populated with the data.
app.config.update(
    DROPZONE_MAX_FILE_SIZE = 1024,
    DROPZONE_TIMEOUT = 5*60*1000)

dropzone = Dropzone(app)
# @app.route('/dragdrop',methods=['POST'])
# def drag_drop():
#     if request.method == 'POST':
#         f = request.files.get('file')
#         filename = secure_filename(f.filename)
#         f.save(os.path.join(app.config['UPLOAD_FOLDER']+ filename))
#         path= os.path.join(app.config['UPLOAD_FOLDER']+ filename)
#         pageDOCX = textract.process(path)
#         pageDOCX=str(pageDOCX,'utf-8')
#         pageDOCX= pageDOCX.replace('\n',' ').strip()
        
#     return render_template('index.html')

@app.route('/dragdrop',methods=['POST', 'GET'])
def drag_drop():
    if request.method == 'POST':
        my_files = request.files    
        title = request.form['title'] 

        for item in my_files:
            uploaded_file = my_files.get(item)
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER']+ filename))
            path= os.path.join(app.config['UPLOAD_FOLDER']+ filename)
            pageDOCX = textract.process(path)
            pageDOCX=str(pageDOCX,'utf-8')
            pageDOCX= pageDOCX.replace('\n',' ').strip() 
            
            pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
            matches = pattern.findall(pageDOCX)
            pattern1 = re.compile(
                r'\d{3}[-\.\s]??\d{4}[-\.\s]??\d{4}|\d{5}[-\.\s]??\d{3}[-\.\s]??\d{3}|(?:\d{4}\)?[\s-]?\d{3}[\s-]?\d{4})')
            matches1 = pattern1.findall(pageDOCX)

            #model call here for getting entities back
            nltk_result= nltk_extraction(pageDOCX)

            # using model_api to get model predictions
            res = model_api(pageDOCX)
            post = Post(title=title, content=pageDOCX, author=current_user, email=str(matches), number=str(matches1), entities= nltk_result, entity=str(res))
            db.session.add(post)
            db.session.commit()
        
        flash('Your posts has been uploaded!', 'success')
        return redirect(url_for('home')) 
        
    return render_template('index.html')
