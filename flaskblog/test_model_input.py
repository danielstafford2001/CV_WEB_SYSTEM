import tensorflow as tf
from model import text_preprocessing


text = """

Daniel Stafford
07926669340 | danielstafford@me.com

 EDUCATION	
 
University
Royal Holloway University of London, BSc Computer Science and Artificial Intelligence 1st Honors in both first and second years
Won first prize for the best second year group project, focused on software engineering.
Sixth Form
A levels: Mathematics, Economics, French
 
UK, London
2019-2022


UK, London
2017-2019 

 WORK EXPERIENCE	 
Inverewe Capital London Limited
  Hedge fund internship (2 months)
•	Completed a full stack trade processing system using Python. This required both Bloomberg and Linedata LGH (Portfolio Management System). The project enabled traders to input their trade information for automatic uploading, saving time and increasing the flow of data.
•	Completed daily reports on P&L, where my understanding of trading strategies was tested each day. For example, the fund had multiple option trades, so I would use payoff diagrams and other tools to predict potential reasons for them. My understanding of traders’ adjustments enabled me to provide daily portfolio summaries to investors, which improved their understanding of the funds economic outlook.
•	Created a Python script to retrieve portfolio quantities from the backend to inform investors of portfolio holdings. This resulted in closer liaison between investors and the fund.
•	Following this internship, Inverewe employed me on a freelance basis, to complete multiple quantitative and qualitative projects for the future.

Oilprice.com
   Oil and commodities internship (2 months)
•	Worked on a machine learning program for crude oil (WTI/Brent) where various data inputs were submitted using API’s such as: commodity prices, rig counts, refining/chemical prices and more. This helped to predict future price movements, using Tensorflow and incorporating multiple sequence modelling algorithms & architectures.
•	Researched existing client database to create a shortlist of candidates for an upgrade to the premium services based on previous purchasing history and their activity on Oilprice.com website.
-	Telemarketed and met with a number of decision makers in the shortlisted group, this resulted in 26 customers agreeing to the upgrade.
•	Helped analysts with researching companies and market opportunities in the  Energy Tech and ESG space. I used Python to make calls to the FRED API in order to create multi-linear regression models to find complex relationships, newly discovered to the team.

Chelsfield
Property Investment insight week
•	Analysed hotel investments and graded them in terms of yield with supplied data.
-	Researched data and created corresponding graphs and charts and added all information to a SQL database for further relational findings.
-	Reduced shortlist of potential purchases by 70% by defining specific database queries to narrow down the choices. Later, presented these findings to the Group CEO justifying my conclusions.
•	Identified potential tenants for vacant retail units in Knightsbridge.
-	Researched their financial standing, brand and potential long-term expansion.
-	Finally, identified a nucleus of favored companies to approach.

BNP Paribas
Investment banking insight week
•	Was exposed to the trading floor where I gained familiarization into Bloomberg Terminal.
•	Shared a desk with the fixed income team where I learnt about bond fundamentals.
 
   UK, London
May-Jul 2021

Mexico, City
 Dec 2019

UK, London
Jun-Sep 2019

UK, London
July 2018
 

 INTERESTS / EXTRA- CURRICULAR ACTIVITIES	
  Programming Languages: Python, Tensorflow, HTML, CSS, C, Java, SQL, PostgreSQL.
  Other skills: Git, Javadoc, SVN, Maven and Test-Driven Development.
  Languages: English & French Bilingual
  Interests:
-	Golf. I have represented Surrey County in the past and currently represent my university’s team. I achieved a golf scholarship at Wentworth Golf Club.
-	University Computer Science Society, helping new programmers and aiding other students in their personal projects.
-	Developing a Deep Learning algorithm within the field of Natural Language Processing, focusing on stream-lining recruitment through a web app. I head a group of 3 in this development.

"""

#print(text)

new_model = tf.keras.models.load_model('my_model.h5')
#x_train, x_test, y_train, y_test, words, tags, max_len,X,y, num_words,num_tags = text_preprocessing()