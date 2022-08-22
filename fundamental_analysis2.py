import streamlit as st
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.style as style
from datetime import date
import matplotlib.dates as dates
from matplotlib.dates import MonthLocator, DateFormatter, WeekdayLocator
from matplotlib.ticker import NullFormatter
import seaborn as sns
from urllib.request import urlopen
import json 
from pandas.io.json import json_normalize
import pandas as pd
import requests
from matplotlib.figure import Figure
from bs4 import BeautifulSoup
import requests
import bs4
import numpy as np
from urllib.request import urlopen
import yfinance as yf
import datetime
from scipy import stats
import base64
from io import BytesIO
import numpy
import time
import hashlib



st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)





def color_negative_red(val):
	"""
	Takes a scalar and returns a string with
	the css property `'color: red'` for negative
	strings, black otherwise.
	"""
	color = 'red' if val < 0 else 'black'
	return 'color: %s' % color



html_code='''
<div style="background-color:#464e5f;padding:15px;border-radius:8px;margin:1px;">
	<h1 style="color:white;text-align:center;font-size:50px;">{}</h1>
		</div>
'''

overview_style='''
<div style="background-color:#ada397;padding:3px;border-radius:8px;margin:2px;">
	<h4 style="color:black;text-align:center;font-size:20px;">{}</h4>
		</div>
'''

sidebar_title='''
<div style>
	<h1 style="color:#CD5C5C;text-align:center;font-size:35px;">{}</h1>
		</div>
'''
title_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<h6>Author:{}</h6>
	<br/>
	<br/>	
	<p style="text-align:justify">{}</p>
	</div>
	"""
article_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<h6>Author:{}</h6> 
	<h6>Post Date: {}</h6>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>
	<p style="text-align:justify">{}</p>
	</div>
	"""
head_message_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">Title: {}</h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;">
	<h4 style="color:white;text-align:center;">Company Name: {}</h1>		
	<h4 style="color:white;text-align:center;">Date: {}</h1>	
	</div>
	"""
full_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<p style="text-align:justify;color:black;padding:10px">{}</p>
	</div>
	"""


def create_ratios(pl,bl,cfs,multiselect,str_val):
	pl=pl.transpose()
	reset_names=["Sales\xa0+","Expenses\xa0+",'Operating Profit','OPM%','Other Income','Profit before tax','Net Profit']
	l1=['sales','expenses','opt','opm','oi','pbt','pat']
	x=0
	for i in reset_names:
		pl.rename(columns={i:l1[x]},inplace=True)
		x=x+1
	
	change_datatype=['sales','expenses','opt','oi','Interest','Depreciation','pbt','pat']

	for i in change_datatype:
		pl[i] = pd.to_numeric(pl[i])

	pl.rename(columns={'OPM %':'opm'},inplace=True)
	bl=bl.transpose()
	b_reset_names=['Share Capital\xa0+','Other Liabilities\xa0+','Total Liabilities','Fixed Assets\xa0+','CWIP','Other Assets\xa0+','Total Assets']
	new_names=['sharecap','otl','tl','nfa','cwip','oa','ta']	
	for i in range(7):
		bl.rename(columns={b_reset_names[i]:new_names[i]},inplace=True)
	bl.rename(columns={'ol':'otl'},inplace=True)
	cfs=cfs.transpose()
	cfs_reset_names=['Cash from Operating Activity\xa0+','Cash from Investing Activity\xa0+','Cash from Financing Activity\xa0+','Net Cash Flow']
	cf_new_names=['cfo','cfi','cff','ncf']
	for i in range(4):

		cfs.rename(columns={cfs_reset_names[i]:cf_new_names[i]},inplace=True)

	interest_covg= pl.opt/pl.Interest
	interest_covg=interest_covg.round(2)
	if 'interest_covg' in multiselect:
		return interest_covg[-1]
	bl.rename(columns={'sharecapital':'sharecap'},inplace=True)

	debt_equity=(bl.Borrowings)/(bl.sharecap+bl.Reserves)
	debt_equity=debt_equity.round(2)

	if 'Debt/Equity' in multiselect:
		return debt_equity[-1]




	workingcap_sales=(bl.oa-bl.otl)/pl.sales
	workingcap_sales=(workingcap_sales*100).round(2)
	workingcap_sales=((workingcap_sales).apply(str))+'%'


	indx=list((interest_covg.index))

	s1 = pd.Series(list(interest_covg), index=list((interest_covg.index)), name='interest_covg')
	s2 = pd.Series(list(debt_equity), index=list((debt_equity.index)), name='debt_equity')
	
	s3 = pd.Series(list(workingcap_sales), index=list((workingcap_sales.index)), name='workingcap_sales')

	df_stability=pd.concat([s1,s2,s3], axis=1).transpose()

	#Margins
	opm= (((pl.opt/pl.sales)*100).round(2)).apply(str)+'%'
	npm= (((pl.pat/pl.sales)*100).round(2)).apply(str)+'%'
	pbt=(((pl.pbt/pl.sales)*100).round(2)).apply(str)+'%'
	ebd=(((pl.pbt-pl.Depreciation/pl.sales)*100).round(2)).apply(str)+'%'


	indx=list(pl.index)
	x1 = pd.Series(list(opm), index=list((opm.index)), name='OPM%')
	x2 = pd.Series(list(pbt), index=list((pbt.index)), name='PBT%')
	x3 = pd.Series(list(npm), index=list((npm.index)), name='NPM%')

	df_margins=pd.concat([x1, x2,x3], axis=1).transpose()



#Duponts

	nfat=(pl.sales/bl.nfa).round(2)

	lvg=(bl.ta/(bl.sharecap+bl.Reserves)).round(2)
	


	roe=(((nfat* lvg * (pl.pat/pl.sales))*100).round(2).apply(str) )+'%'

	indx=list(pl.index)
	x1 = pd.Series(list(nfat), index=list((nfat.index)), name='Asset_Turnover')
	x2 = pd.Series(lvg, index=list((lvg.index)), name='Leverage_ratio')
	x3 = pd.Series(list(npm), index=list((npm.index)), name='NPM%')
	x4 = pd.Series(list(roe), index=list((roe.index)), name='ROE%')

	if 'roe%' in multiselect:
		return roe[-2]

	df_dupont=pd.concat([x1, x2,x3,x4], axis=1).transpose()


	

#Yearly growth rate
#Yearly growth rate

	sales=[]
	pbt=[]
	opt=[]
	pat=[]
	indx=list(pl.index)
	for i in range(1,len(indx)):
	  sales.append(str((((pl.sales[i]/pl.sales[i-1])-1)*100).round(2))+'%')
	  opt.append(str((((pl.opt[i]/pl.opt[i-1])-1)*100).round(2))+'%')
	  pbt.append(str((((pl.pbt[i]/pl.pbt[i-1])-1)*100).round(2))+'%')
	  pat.append(str((((pl.pat[i]/pl.pat[i-1]-1))*100).round(2))+'%')
	indx.pop()

	x1 = pd.Series(sales, index=indx, name='Sales_growth')
	x2 = pd.Series(pbt, index=indx, name='PBT_growth')
	x3 = pd.Series(pat, index=indx, name='PAT_growth') 
	x4= pd.Series(opt, index=indx, name='OPT_growth')


	
	df_ygrowth=pd.concat([x1,x4, x2,x3], axis=1).transpose()


	

	cagr_indx=['10 Years', ' 5 Years', '3 Years']

#10 year CAGR

	s_cagr10= (pl.sales.loc['Mar 2020']/pl.sales.loc['Mar 2010'])**(1/10)-1
	opt_cagr10= (pl.opt.loc['Mar 2020']/pl.opt.loc['Mar 2010'])**(1/10)-1
	pbt_cagr10= (pl.pbt.loc['Mar 2020']/pl.pbt.loc['Mar 2010'])**(1/10)-1
	pat_cagr10= (pl.pat.loc['Mar 2020']/pl.pat.loc['Mar 2010'])**(1/10)-1

	s_cagr10=str((s_cagr10*100).round(2))+'%'
	opt_cagr10=str((opt_cagr10*100).round(2))+'%'
	pbt_cagr10=str((pbt_cagr10*100).round(2))+'%'
	pat_cagr10=str((pat_cagr10*100).round(2))+'%'

	#5 year CAGR
	s_cagr5= (pl.sales.loc['Mar 2020']/pl.sales.loc['Mar 2015'])**(1/5)-1
	opt_cagr5= (pl.opt.loc['Mar 2020']/pl.opt.loc['Mar 2015'])**(1/5)-1
	pbt_cagr5= (pl.pbt.loc['Mar 2020']/pl.pbt.loc['Mar 2015'])**(1/5)-1
	pat_cagr5= (pl.pat.loc['Mar 2020']/pl.pat.loc['Mar 2015'])**(1/5)-1

	s_cagr5=str((s_cagr5*100).round(2))+'%'
	opt_cagr5=str((opt_cagr5*100).round(2))+'%'

	pbt_cagr5=str((pbt_cagr5*100).round(2))+'%'
	pat_cagr5=str((pat_cagr5*100).round(2))+'%'

	#3 year CAGR
	s_cagr3= (pl.sales.loc['Mar 2020']/pl.sales.loc['Mar 2017'])**(1/3)-1
	opt_cagr3= (pl.opt.loc['Mar 2020']/pl.opt.loc['Mar 2017'])**(1/3)-1
	pbt_cagr3= (pl.pbt.loc['Mar 2020']/pl.pbt.loc['Mar 2017'])**(1/3)-1
	pat_cagr3= (pl.pat.loc['Mar 2020']/pl.pat.loc['Mar 2017'])**(1/3)-1

	s_cagr3=str((s_cagr3*100).round(2))+'%'
	opt_cagr3=str((opt_cagr3*100).round(2))+'%'
	pbt_cagr3=str((pbt_cagr3*100).round(2))+'%'
	pat_cagr3=str((pat_cagr3*100).round(2))+'%'

	z1 = pd.Series([s_cagr10,s_cagr5,s_cagr3], index=cagr_indx, name='Sales Growth%')
	z2 = pd.Series([opt_cagr10,opt_cagr5,opt_cagr3], index=cagr_indx, name='OPT Growth%')
	z3 = pd.Series([pbt_cagr10,pbt_cagr5,pbt_cagr3], index=cagr_indx, name='PBT Growth%')
	z4 = pd.Series([pat_cagr10,pat_cagr5,pat_cagr3], index=cagr_indx, name='PAT Growth%')



	
	df_cagr =pd.concat([z1, z2,z3,z4], axis=1).transpose()
	
	# Capital Allocation
	
	indx=list(pl.index)
	nfat= (pl.sales/bl.nfa).round(2)
	rofa=(((pl.pat/bl.nfa)*100).round(2)).apply(str)+'%'
	debt=[]
	assets=[]
	for i in range(1,len(indx)-1):
	  debt.append(str((((bl.Borrowings[i]/bl.Borrowings[i-1])-1)*100).round(2))+'%')
	  assets.append(str((((bl.nfa[i]/bl.nfa[i-1])-1)*100).round(2))+'%')



	indx.pop(0)
	indx.pop()
	u3 = pd.Series(debt, index=indx, name='Borrowing Incr/Dcr%')
	u4 = pd.Series(assets, index=indx, name='Assets Incr/Dcr%')

	df_comparison=pd.concat([u3,u4], axis=1).transpose()
	

	#All 
	indx=list(pl.index)
	v1 = pd.Series(list(pl.pat), index=indx, name='PAT')
	v2 = pd.Series(list(cfs.cfo)+[None], index=indx, name='CFO')

	df_cfovpat=pd.concat([v1,v2],axis=1).transpose()
	

	capex=[]
	
	for i in range(1,len(bl.index)):
	  capex.append((bl.nfa[i]+bl.cwip[i])-(bl.nfa[i-1]+bl.cwip[i-1]))


	fcf=[]
	cfo=list(cfs.cfo)[0:]
	for i in range(0,len(capex)):
	  fcf.append(cfo[i]-capex[i])




	

	indx=list(cfs.index)[0:]
	l1 = pd.Series(cfo ,index=indx, name='CFO')
	l2 = pd.Series(capex, index=indx, name='CAPEX')
	l3 = pd.Series(fcf, index=indx, name='FCF')


	df_fcf=pd.concat([l1,l2,l3],axis=1).transpose()
	

	l1=[sum(df_fcf.transpose().CFO),sum(df_fcf.transpose().CFO.loc['Mar 2015':]),sum(df_fcf.transpose().CFO.loc['Mar 2017':])]
	l2=[sum(df_fcf.transpose().CAPEX),sum(df_fcf.transpose().CAPEX.loc['Mar 2015':]),sum(df_fcf.transpose().CAPEX.loc['Mar 2017':])]
	l3=[sum(df_fcf.transpose().FCF),sum(df_fcf.transpose().FCF.loc['Mar 2015':]),sum(df_fcf.transpose().FCF.loc['Mar 2017':])]
	l4=[((l3[0]/l1[0])*100 ),((l3[1]/l1[1])*100 ),((l3[2]/l1[2])*100 ) ]
	g1=pd.Series(l1,index=cagr_indx,name='CFO')
	g2=pd.Series(l2,index=cagr_indx,name='CAPEX')     
	g3=pd.Series(l3,index=cagr_indx,name='FCF')
	g4=pd.Series(l4,index=cagr_indx,name='FCF/CFO%').round(2).apply(str)+'%' 
	df_fcf_cagr=pd.concat([g1,g2,g3,g4],axis=1).transpose() 

	if 'fcf' in multiselect:
		return df_fcf_cagr.iloc[3,0]       
	                                           

	k1=[sum(cfs.cfo),sum(cfs.cfo.loc['Mar 2015':]),sum(cfs.cfo.loc['Mar 2017':])]
	k2=[sum(df_cfovpat.transpose().PAT),sum(df_cfovpat.transpose().PAT.loc['Mar 2015':]),sum(df_cfovpat.transpose().PAT.loc['Mar 2017':])]
	c1=pd.Series(k1,index=cagr_indx,name='CFO')
	c2=pd.Series(k2,index=cagr_indx,name='PAT')
	df_cfovpat_cagr=pd.concat([c1,c2],axis=1).transpose()     
	

	

	df_analysis_cagr=pd.concat([df_cagr,df_fcf_cagr,df_cfovpat_cagr],axis=0,keys=['Growth Rates','Free Cash Flow','Working Capital'])
	





	df_analysis=pd.concat([df_margins,df_ygrowth,df_stability,df_dupont,df_comparison,df_cfovpat,df_fcf],axis=0,keys=['Margins','Growth Rates','Debt & Stability','DuPonts Analysis','Return On Assets','Assets vs Debt','CFO vs PAT','FCF'])
	df_analysis.drop(columns=['Mar 2009','TTM'],inplace=True)

	if str_val=='Ratio': 
		
		if 'Growth Rates' in multiselect:
			st.markdown('## Growth Rates')
			st.dataframe(df_ygrowth)
			st.dataframe(df_cagr)
			st.markdown(overview_style.format(get_table_download_link(df_cagr)),unsafe_allow_html=True)
		if 'Margins' in multiselect:
			st.markdown('## Margins')
			st.dataframe(df_margins)
			st.markdown(overview_style.format(get_table_download_link(df_margins)),unsafe_allow_html=True)

		if 'Debt & Stability' in multiselect:
			st.markdown('## Debt and Stability')
			st.dataframe(df_stability)

		if 'Duponts Analysis' in multiselect:
			st.markdown('## Duponts Analysis')
			st.dataframe(df_dupont)

		if 'Assets vs Debt' in multiselect:
			st.markdown('## Assets vs Debt')
			st.dataframe(df_comparison)

		if 'Cash Conversion' in multiselect:
			st.markdown('## Cash Conversion & Working Capital')
			st.table(df_cfovpat)
			st.dataframe(df_fcf)
			st.dataframe(df_fcf_cagr)
			st.dataframe(df_cfovpat_cagr)

		if 'All' in multiselect:
			st.markdown('## All Ratios')
			st.dataframe(df_analysis_cagr)
			st.markdown(overview_style.format(get_table_download_link(df_analysis_cagr)),unsafe_allow_html=True)

			st.dataframe(df_analysis)
			st.markdown(overview_style.format(get_table_download_link(df_analysis)),unsafe_allow_html=True)

	

	else:





		if 'Profitability' in multiselect:
			x=new_index=[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
			y=pd.concat([pl.sales,pl.opt,pl.pat],axis=1)

			st.markdown('# Sales')

			st.bar_chart(pl.sales,height=600,use_container_width=True)

		sns.set_style("darkgrid", {"axes.facecolor": ".9"})
		sns.set_context("poster")
		if 'Margins Visualization' in multiselect:

			st.set_option('deprecation.showPyplotGlobalUse', False)
			Year = [9,10,11,12,13,14,15,16,17,18,19,20,'TTM']
			opm= list(((pl.opt/pl.sales)*100).round(2))
			pbt=list(((pl.pbt/pl.sales)*100).round(2))
			pat=list(((pl.pat/pl.sales)*100).round(2))

			plt.figure(figsize=(8,4))
			plt.plot(Year, opm, color='red', marker='o'  )
			plt.plot(Year, pbt, color='green', marker='o'  )
			plt.plot(Year, pat, color='blue', marker='o'  )
			plt.title('OPM%, EBT% & NPM% ', fontsize=14)
			plt.xlabel('Year', fontsize=12)
			plt.ylabel('Margin%', fontsize=12)
			plt.grid(True)
			
			st.pyplot()

		if 'Interest Coverage' in multiselect:
			Year = [9,10,11,12,13,14,15,16,17,18,19,20,'TTM']

			st.set_option('deprecation.showPyplotGlobalUse', False)
			int_covg_list = list(interest_covg)

			plt.figure(figsize=(12,6))
			plt.plot(Year, int_covg_list, color='red', marker='o'  )

			plt.title('Operating Profit vs Interest', fontsize=18)
			plt.xlabel('Year', fontsize=16)
			plt.ylabel('Interest Coverage', fontsize=16)
			plt.grid(True)
			
			st.pyplot()
		if 'Debt & Liquidity' in multiselect:
			Year = [9,10,11,12,13,14,15,16,17,18,19,20,'TTM']

			current_ratio=bl.oa/(bl.otl)
			current_ratio=current_ratio.round(2)
			crr=list(current_ratio)+[None]
			d_e= list(debt_equity)
			crr=list(current_ratio)


			plt.figure(figsize=(12,6))
			plt.plot(Year, d_e, color='indigo', marker=None,linewidth=3)
			plt.plot(Year, crr, color='lightsalmon', marker=None,linewidth=3)

			plt.title('Debt & Liquidity', fontsize=18)
			plt.xlabel('Year', fontsize=16)
			plt.ylabel('Ratio', fontsize=16,  )
			plt.grid(False)
			plt.legend(['debt/equit','current ratio'],fontsize=12)
			plt.grid(True)
			

			st.pyplot()
		if 'ROE Distribution' in multiselect:
			

			Year = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,'TTM']

			y = numpy.row_stack((list(nfat),list(lvg),pat))
			x = numpy.array(Year)

			# Make new array consisting of fractions of column-totals,
			# using .astype(float) to avoid integer division
			percent = y /  y.sum(axis=0).astype(float) * 100 

			fig = plt.figure(figsize=(12,8))
			ax = fig.add_subplot(111)

			ax.stackplot(x, percent)
			ax.set_title('ROE Composition', )
			ax.set_ylabel('Percent (%)')
			ax.legend(['Asset Turnover','Leverage','Net Profit%'])
			ax.margins(0.01, 0.01)
			st.pyplot(ax)
		
		if 'ROE Comparison' in multiselect:
			roex=np.array(((nfat* lvg * (pl.pat/pl.sales))*100).round(2)) 

	# red dashes, blue squares and green triangles


			Year = [9,10,11,12,13,14,15,16,17,18,19,20,21]
			plt.legend(['ROE%'])
			plt.xlabel('Year')
			plt.ylabel('ROE')
			plt.title('ROE Scatter Plot')
			plt.scatter(Year,roex,marker='o')
			st.pyplot()	
		if 'NFAT' in multiselect:
			barWidth = 0.25
			Year = [9,10,11,12,13,14,15,16,17,18,19,20,'TTM']
			plt.figure(figsize=(15,10))
			# set height of bar
			bars1 = nfat

			 
			# Set position of bar on X axis
			r1 = np.arange(len(bars1))
			r2 = [x + barWidth for x in r1]
			r3 = [x + barWidth for x in r2]
			 
			# Make the plot
			plt.bar(r1, bars1, color='#000080', width=barWidth, edgecolor='black', label='NFAT')

			 
			# Add xticks on the middle of the group bars
			plt.xlabel('group', fontweight='bold')
			plt.xticks([r + barWidth for r in range(len(bars1))], Year)
			 
			# Create legend & Show graphic
			plt.legend(fontsize=12)
			st.pyplot()

		if 'ROA%' in multiselect:
			rofax=(((pl.pat/bl.nfa)*100).round(2))
			Year = [9,10,11,12,13,14,15,16,17,18,19,20,'TTM']


			plt.figure(figsize=(12,8))
			plt.plot(Year, rofax, color='red', marker='o'  )

			plt.title('Return on Assets', fontsize=18)
			plt.xlabel('Year', fontsize=16)
			plt.ylabel('Percentage%', fontsize=16)
			plt.grid(True)
			st.pyplot()





	

	

	


















def multiselect(val2,col2):
	if val2=='Financial Statements':
		return col2.multiselect('',['Income Statement','Balance Sheet','Cash Flow Statement'])
	if val2=='Financial Ratios':
		return col2.multiselect('',['Growth Rates','Margins','Debt & Stability','Duponts Analysis','Return on Assets','All'])

	if val2=='Visualization':
		return col2.multiselect('',['Profitability','Margins Visualization','Interest Coverage','Debt & Liquidity','ROE Comparison','ROE Distribution','NFAT','ROA%','Assets vs Borrowings','CFO vs PAT','Cash Composition'])



def company_info(val,multiselect):
	
	text='screener.in'+val
	url = 'https://google.com/search?q=' + text
	request_result=requests.get( url ) 
	soup = BeautifulSoup(request_result.text,"lxml") 

	s=''
	for i in range(7,len((soup.select('.kCrYT a')[0]).get('href'))):



		if (soup.select('.kCrYT a')[0]).get('href')[i]=='&':

			break
			
		s=s+(soup.select('.kCrYT a')[0]).get('href')[i]


	
	
	if 'Income Statement' in multiselect:
		st.subheader('Income Statement')
		pl=pd.read_html(s)[1]
		
		st.table(pl)
		#st.dataframe(pl.style.applymap(color_negative_red))
		st.markdown(overview_style.format(get_table_download_link(pl)),unsafe_allow_html=True)


		
		#st.markdown(overview_style.format(get_table_download_link(pl)),unsafe_allow_html=True)

	if 'Balance Sheet' in multiselect:
		st.markdown("# Balance Sheet")
		bl=pd.read_html(s)[6]
		st.table(bl)
		
		st.markdown(overview_style.format(get_table_download_link(bl)),unsafe_allow_html=True)
		
	if 'Cash Flow Statement' in multiselect:
		st.markdown("# Cash Flow")

		cfs=pd.read_html(s)[7]
		st.table(cfs)
		st.markdown(overview_style.format(get_table_download_link(cfs)),unsafe_allow_html=True)


	

	
def company_ratios(val,multiselect,str_val):
	text='screener.in'+val
	url = 'https://google.com/search?q=' + text
	request_result=requests.get( url ) 
	soup = BeautifulSoup(request_result.text,"lxml") 

	s=''
	for i in range(7,len((soup.select('.kCrYT a')[0]).get('href'))):



		if (soup.select('.kCrYT a')[0]).get('href')[i]=='&':

			break
			
		s=s+(soup.select('.kCrYT a')[0]).get('href')[i]


	arr_pl=['Mar 2009','Mar 2010','Mar 2011','Mar 2012','Mar 2013','Mar 2014','Mar 2015','Mar 2016','Mar 2017','Mar 2018','Mar 2019','Mar 2020','TTM']
	#arr_bl=['Unnamed: 0','Mar 2009','Mar 2010','Mar 2011','Mar 2012','Mar 2013','Mar 2014','Mar 2015','Mar 2016','Mar 2017','Mar 2018','Mar 2019' 'Mar 2020','TTM']
	arr_cfs=['Mar 2009','Mar 2010','Mar 2011','Mar 2012','Mar 2013','Mar 2014','Mar 2015','Mar 2016','Mar 2017','Mar 2018','Mar 2019','Mar 2020']

	pl=pd.read_html(s)[1].set_index('Unnamed: 0')
	bl=pd.read_html(s)[6].set_index('Unnamed: 0')
	cfs=pd.read_html(s)[7].set_index('Unnamed: 0')
	i=0
	j=0
	k=0
	while pl.shape[1]<13:

		pl['new'+str(i)]=pl.iloc[:,-1]
		i+=1
	while bl.shape[1]<13:
		bl['new'+str(j)]=bl.iloc[:,-1]
		j+=1
	while cfs.shape[1]<12:
		cfs['new'+str(k)]=cfs.iloc[:,-1]
		k+=1

	pl.columns=arr_pl
	bl.columns=arr_pl
	cfs.columns=arr_cfs
	

	create_ratios(pl,bl,cfs,multiselect,str_val)
	


def plot_shareprice(search,date,val,norm,reg):
	st.set_option('deprecation.showPyplotGlobalUse', False)
	

	tickers=[]
	search=search.split(',')
	for search_val in search:
		text='yahoo finance'+search_val+'indian market'+'history'
		url = 'https://google.com/search?q=' + text
		request_result=requests.get( url ) 
		soup = BeautifulSoup(request_result.text,"html.parser") 

		s=''

		for i in range(7,len((soup.select('.kCrYT a')[0]).get('href'))):

			if (soup.select('.kCrYT a')[0]).get('href')[i]=='&':
				break
			s=s+(soup.select('.kCrYT a')[0]).get('href')[i]


		if s[-8:-1]=='history':
			symbol=s[35:-9]
		else:
			symbol=s[35:-1]
		tickers.append(symbol)
	if 'Price Chart' in val:
		if norm==True:

			stocks=yf.download(tickers,start=date,end="2020-01-01")
			stocks=stocks.loc[:,'Close']
			stocks=stocks.div(stocks.iloc[0]).mul(100)
			plt.style.use('seaborn')
			stocks.plot(figsize=(15,8),fontsize=13)
			st.pyplot()
		else:

			
			stocks=yf.download(tickers,start=date,end="2020-01-01")
			stocks=stocks.loc[:,'Close']
			plt.style.use('seaborn')
			stocks.plot(figsize=(15,8),fontsize=13)
			st.set_option('deprecation.showPyplotGlobalUse', False) #Warning for bad written code, change later if required

			st.pyplot()


	if 'Risk vs Return Plot' in val:
		stocks=yf.download(tickers,start=date,end="2020-01-01")
		stocks=stocks.loc[:,'Close']
		pct_return=stocks.pct_change().dropna()
		pct_return.mean()*252## Annualized mean returns

		pct_return.var()*252## Annualized risk
		np.sqrt(pct_return.var()*252)

		pct_return.describe()

		risk_return=yf.download(tickers,start="2012-01-01",end="2020-01-01")

		risk_return=risk_return.loc[:,"Close"].pct_change().dropna()

		risk_return=risk_return.describe().loc[['mean','std']]
		st.dataframe(risk_return)

		risk_return.loc['mean']=risk_return.loc['mean'].mul(252)

		risk_return.loc['std']=risk_return.loc['std'].mul(np.sqrt(252))

		risk_return=risk_return.transpose()

		risk_return.plot(kind = "scatter", x = "std", y = "mean", figsize = (15,12), s = 50, fontsize = 15)
		for i in risk_return.index:
		    plt.annotate(i, xy=(risk_return.loc[i, "std"]+0.002, risk_return.loc[i, "mean"]+0.002), size = 15)
		plt.xlabel("ann. Risk(std)", fontsize = 15)
		plt.ylabel("ann. Return", fontsize = 15)
		plt.title("Risk/Return", fontsize = 20)
		plt.show()
		st.pyplot()

	if 'Correlation Heatmap' in val:
		correlation=yf.download(tickers,start=date,end="2020-01-01")
		correlation=correlation.loc[:,'Close']


		correlation.corr()


		plt.figure(figsize=(12,8))
		sns.set(font_scale=1.4)
		sns.heatmap(correlation.corr(), cmap = "Reds", annot = True, annot_kws={"size":10})
		plt.show()
		st.pyplot()


	if 'Linear Regression Plot and Prediction' in val:

		pat_final=[]
		capex_final=[]
		names=[]
		eps=[]
    
		for j in search:
			text='screener.in'+ j
			url = 'https://google.com/search?q=' + text
			request_result=requests.get( url ) 
			soup = BeautifulSoup(request_result.text,"lxml") 

			s=''
			for i in range(7,len((soup.select('.kCrYT a')[0]).get('href'))):
				if (soup.select('.kCrYT a')[0]).get('href')[i]=='&':
					break
				s=s+(soup.select('.kCrYT a')[0]).get('href')[i]
			pl=pd.read_html(s)[1].set_index('Unnamed: 0')
			bl=pd.read_html(s)[6].set_index('Unnamed: 0')
			cfs=pd.read_html(s)[7].set_index('Unnamed: 0')
			eps.append(float(pl.iloc[:,-1][-2]))


			pat=list(pd.to_numeric(pl.loc['Net Profit']))
			capex=list(bl.loc['Investments']+bl.loc['CWIP']+bl.loc["Fixed Assets\xa0+"])
			pat_final=pat_final+pat
			capex_final=capex_final+capex
			names=names+[j for x in range(0,len(pat))]
			#slope, intercept, r_value, pv, se = stats.linregress(pat,capex)
			#st.write(j.upper()+' :			Slope: '+str(slope)+ '									Intercept: '+str(intercept))



		df=pd.DataFrame(list(zip(pat_final, capex_final,names)),columns =['Pat', 'Capex','Company'])
		if reg==True:
			sns.lmplot(data=df, x='Capex',y='Pat',hue='Company')
		else:
			sns.lmplot(data=df, x='Capex',y='Pat',col='Company')
		plt.show()
		st.pyplot()
		st.set_option('deprecation.showPyplotGlobalUse', False)
		slope, intercept, r_value, pv, se = stats.linregress(df['Pat'],df['Capex'])

		sns.regplot(x="Pat", y="Capex", data=df, ci=None, label="y={0:.1f}x+{1:.1f}".format(slope, intercept)).legend(loc="best")

		plt.show()
		st.pyplot()
		pat_cagr=[]
		current_pe=[] #ticker price and EPS
		cmmp=[]
		forward_pe=[]
		forward_eps=[]
		roi=[]

			
		stocks=yf.download(tickers,start='2021-04-22',end="2021-05-01")
		stocks=stocks.loc[:,'Close']

		for i in range(0,len(search)):

			df_search=df[df['Company']==search[i]]
			a=((((df_search.iloc[-1][0]/df_search.iloc[0][0])**(1/12))-1)*100).round(2)
			b=(stocks.iloc[-1][tickers[i]]/eps[i]).round(2)
			c=stocks.iloc[-1][tickers[i]]
			d= b*(((a*30/100)+100)/100)
			e=eps[i]*((100+a)/100)
			f= ((d*e/c-1).round(2))*100
			pat_cagr.append(str(a)+'%')
			current_pe.append(b)
			cmmp.append(c)
			forward_pe.append(d)
			forward_eps.append(e)
			roi.append(str(f)+'%')
		final_roi_pred=pd.DataFrame(list(zip(pat_cagr,current_pe,cmmp,forward_pe,forward_eps,roi)), columns =['CAGR% of PAT', 'Current PE','Market Price','Forward PE','Forwards EPS','Return on Investment'],index=search)
		st.dataframe(final_roi_pred.round(2))

def company_industry_search(industry,pe='10,50',mcap='500,20000',cmmp='100,3000',roce='10,25'):
	#pe=list(pd.to_numeric(pe.split(',')))
	#mcap=[int(x) for x in mcap.split(',')]
	#cmmp=[int(x) for x in cmmp.split(',')]
	#roce=[int(x) for x in roce.split(',')]

		
	search_val=industry+'comapnies list'
	text='screener.in'+search_val
	url = 'https://google.com/search?q=' + text
	request_result=requests.get( url ) 
	soup = BeautifulSoup(request_result.text, 
	                     "lxml") 


	# Iterate through the object  
	# and print it as a string. 
	s=''
	for i in range(7,len((soup.select('.kCrYT a')[0]).get('href'))):
		if (soup.select('.kCrYT a')[0]).get('href')[i]=='&':
			break
		s=s+(soup.select('.kCrYT a')[0]).get('href')[i]

	df=pd.read_html(s+'?page=1')[0]
	for i in range(2,15):
		if pd.read_html(s+'?page='+str(i))[0].equals(pd.read_html(s+'?page=1')[0]):
			break
		df=pd.concat([df,pd.read_html(s+'?page='+str(i))[0]],axis=0)

	company_list=df.reset_index(drop=True).drop(columns='S.No.')
	
    
	
	company_list=company_list.drop(company_list.columns[[5,6,7,8]],axis=1)
	cols = company_list.columns
	company_list[cols[1:]] =company_list[cols[1:]].apply(pd.to_numeric, errors='coerce')

	
	#Name  CMP  Rs.    P/E  Mar Cap  Rs.Cr.  Div Yld  % ROCE  %

	#company_list.reset_index(drop=True,inplace=True)

	#final_company_list=company_list[(company_list['P/E']>pe[0]) & (company_list['P/E']<pe[1])]
	final_company_list=company_list[((company_list['P/E']>5) & (company_list['P/E']<20)) & ((company_list['Mar Cap  Rs.Cr.']>1000) & (company_list['Mar Cap  Rs.Cr.']<1000000)) & ((company_list['CMP  Rs.']>200) & (company_list['CMP  Rs.']<3000)) & ((company_list['ROCE  %']>5) & (company_list['ROCE  %']<30))]
	
	final_company_list.reset_index(drop=True,inplace=True)
	st.dataframe(final_company_list)


























def to_excel(df):
	output = BytesIO()
	writer = pd.ExcelWriter(output, engine='xlsxwriter')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()
	processed_data = output.getvalue()
	
	return processed_data

def get_table_download_link(df):
	"""Generates a link allowing the data in a given panda dataframe to be downloaded
	in:  dataframe
	out: href string
	"""
	val = to_excel(df)
	b64 = base64.b64encode(val)  # val looks like b'...'
	
	return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download Excel file</a>' # decode b'abc' => abc



def func():
	st.markdown(html_code.format('Financial Analysis'),unsafe_allow_html=True)
	st.write("")
	st.write("")
	st.write("")
		
	search_val=st.text_input('Search Company')

		

		
		
	
		
	val=st.multiselect('',['Income Statement','Balance Sheet','Cash Flow Statement'])
	if st.button('Search'):
		company_info(search_val,val)


def main():
	
	st.sidebar.subheader('Investico')
	prompts=st.sidebar.multiselect('Select Prompts',['Intruction','Tips','Explanation'])
	#st.write("")
	
	st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
	)

	st.markdown(
	    """
	<style>
	.sidebar .sidebar-content {
	    background-image:#ffe6e6;
	    color:#ffe6e6;
	    background-color:#ffe6e6;
	}
	</style>
	""",
	unsafe_allow_html=True,
	)


	option=st.sidebar.radio('Main Menu',['Home','Financial Statements','Ratio Analysis','Visualization','Valuation Analysis',"Member's Home",'About'])
	


	if option=='Home':
	
		st.write("")		
		st.write("")
		st.write("")
		st.write("## Search Criteria ")
		st.write("")
		col1,col2,col3,col4=st.beta_columns(4)
		pe=col1.text_input('Price to Earnings')
		mcap=col2.text_input('Market Capitalization')
		cmmp=col3.text_input('Current Market Price')
		roce=col4.text_input('Return on Capital Employed')
		industry_list=['Agro Chemicals','Air Transport Service','Alcoholic Beverages','Auto Ancillaries','Auto Mobile','Banks','Bearings','Cables','Capital Goods- Electrical Equipments','Capital Goods- Electrical Equipments','Castings, Forgings and Fastners','Cement','Ceramic-Products','Chemicals','Computer Education','Construction','Credit Rating Agencies','Crude Oil and Natural Gas','Jewellery','Dry Cells','E-commerce','Edible Oil','Education','Electronics','Engineering','Entertainment','ETF','Ferro Alloys','Fertilizers','Finance','FMCG','Gas Distribution','Glass Products','Healthcare','Hotels  Restaurants','Infrastructure Developers','Infrastructure Investment Trusts','Insurance','IT-Hardware','IT-Software','Leather','Logistics','Marine Port and Services','Media-Print/Television/Radio','Mining and Mineral Products','Miscellaneous']+['Non Ferrous Metal','Oil Drill','Online Media','Packaging','Paints/Varnish','Paper','Petro Chemicals','Pharmaceutical','Plastic Products','Power Generation and Distribution','Power Infrastructure','Printing and Stationery','Quick Service Restaurant','Railways','Readymade Garments/Apparells','Real Estate Investment Trusts','Realty','Refineries','Retail','Sanitary Ware','Ship Building','Shipping','Stell','Stock/Commodity Brokers','Sugar','Telecom-Handset/Mobile','Telecom Service','Textile','Tobacco Products','Trading','Tyres']
		industry_val=st.selectbox('Select Industry',industry_list)

		if st.checkbox('Search'):
			company_industry_search(industry_val,pe,mcap,cmmp,roce)
				

		


	if option=='Financial Statements':
	

		st.markdown(html_code.format('Financial Analysis'),unsafe_allow_html=True)
		st.write("")
		st.write("")
		st.write("")
			
		search_val=st.text_input('Search Company')
	
			

			
			
		
			
		val=st.multiselect('',['Income Statement','Balance Sheet','Cash Flow Statement'])
		if st.button('Search'):


				#my_bar = st.progress(0)
				#for percent_complete in range(10):
					#time.sleep(0.1)
					#my_bar.progress(percent_complete + 1)
				strk='    '.join([str(elem) for elem in val])

				with st.spinner('Loading.................................'):
					time.sleep(2)
					st.success('Displaying  : '+strk)

				company_info(search_val,val)
		if 'Intruction' in prompts:
			
			st.info("Income statement – \n An income statement is a financial statement that shows you the company's income and expenditures. It also shows whether a company is making profit or loss for a given period.")
			st.info("Balance Sheet-A balance sheet is a financial statement that reports a company's assets, liabilities and shareholders' equity at a specific point in time, and provides a basis for computing rates of return and evaluating its capital structure.")
			st.info("A cash flow statement is a financial statement that summarizes the amount of cash and cash equivalents entering and leaving a company. The cash flow statement measures how well a company manages its cash position, meaning how well the company generates cash to pay its debt obligations and fund its operating expenses.")

	if option=='Ratio Analysis':
		
		st.markdown(html_code.format('Ratio Analysis'),unsafe_allow_html=True)
		st.write("")		
		st.write("")
		st.write("")
		search_val=st.text_input('Search Company')
		val=st.multiselect('',['Growth Rates','Margins','Debt & Stability','Duponts Analysis','Return on Assets','All'])
		if st.button('Search'):
			strk='    '.join([str(elem) for elem in val])

			with st.spinner('Loading.................................'):
				time.sleep(2)
				st.success('Displaying  : '+strk)
			company_ratios(search_val,val,'Ratio')
		if 'Intruction' in prompts:
			st.info("Growth rates refer to the percentage change of a specific variable within a specific time period. For investors, growth rates typically represent the compounded annualized rate of growth of a company's revenues, earnings, dividends, or even macro concepts, such as gross domestic product (GDP) and retail sales.")
			st.info("Margin is the money borrowed from a brokerage firm to purchase an investment. It is the difference between the total value of securities held in an investor's account and the loan amount from the broker. Buying on margin is the act of borrowing money to buy securities.")
			st.info("Debt means the amount of money which needs to be repaid back and financing means providing funds to be used in business activitiesFinancial stability is defined in terms of its ability to facilitate and enhance economic processes, manage risks, and absorb shocks. Moreover, financial stability is considered a continuum: changeable over time and consistent with multiple combinations of the constituent elements of finance.")
			st.info("A DuPont analysis is used to evaluate the component parts of a company's return on equity (ROE). This allows an investor to determine what financial activities are contributing the most to the changes in ROE. An investor can use analysis like this to compare the operational efficiency of two similar firms.")
			st.info("Return on assets (ROA) is an indicator of how profitable a company is relative to its total assets. ROA gives a manager, investor, or analyst an idea as to how efficient a company's management is at using its assets to generate earnings.")
			st.info("Profitability is a measurement of efficiency – and ultimately its success or failure. A further definition of profitability is a business's ability to produce a return on an investment based on its resources in comparison with an alternative investment")
			st.info("The interest coverage ratio is a debt and profitability ratio used to determine how easily a company can pay interest on its outstanding debt. The interest coverage ratio may be calculated by dividing a company's earnings before interest and taxes (EBIT) by its interest expense during a given period.")
			st.info("Debt means the amount of money which needs to be repaid back and financing means providing funds to be used in business activities Liquidity refers to the ease with which an asset, or security, can be converted into ready cash without affecting its market price. Cash is the most liquid of assets while tangible items are less liquid. The two main types of liquidity include market liquidity and accounting liquidity.")
			st.info("Net Fixed-asset turnover is the ratio of sales to the value of fixed assets. It indicates how well the business is using its fixed assets to generate sales")
			
	if option=='Visualization':
		
		st.markdown(html_code.format('Visualization'),unsafe_allow_html=True)
		st.write("")		
		st.write("")
		st.write("")
		search_val=st.text_input('Search Company')
		val=st.multiselect('',['Margins Visualization','Profitability','Interest Coverage','Debt & Liquidity','ROE Distribution','ROE Comparison','NFAT','ROA%'])
		if st.button('Search'):
			strk='    '.join([str(elem) for elem in val])

			with st.spinner('.............................Loading.................................'):
				time.sleep(2)
				st.success('Displaying  : '+strk)
			company_ratios(search_val,val,'Visualization')


	if option=='Valuation Analysis':
		st.markdown(html_code.format('Valuation Analysis'),unsafe_allow_html=True)
		st.write("")		
		st.write("")
		st.write("")
		
		list_val=[]
		
		norm=False
		reg=False
		price_val=st.text_input('Valuate')
		val=st.multiselect('',['Price Chart','Risk vs Return Plot','Correlation Heatmap','PAT vs Capex Plot','Linear Regression Plot and Prediction'])
		date = st.date_input('start date', datetime.date(2011,1,1))
		if st.checkbox('Normalized Price'):
			norm=True
		if st.checkbox('Gathered Regression Plot'):
			reg=True





		if st.button('Search'):
			strk='    '.join([str(elem) for elem in val])

			with st.spinner('............................................Loading.........................................'):
				time.sleep(2)
				st.success('Displaying  : '+strk)
			
			plot_shareprice(price_val,date,val,norm,reg)
		if "Instruction" in prompts:
			st.info("A price chart is a sequence of prices plotted over a specific timeframe. In statistical terms, charts are referred to as time series plots. On the chart, the y-axis (vertical axis) represents the price scale and the x-axis (horizontal axis) represents the time scale. Prices are plotted from left to right across the x-axis, with the most recent plot being the furthest right")
			st.info("A Risk-Return Plot is a graph depicting portfolio or stock risk on the x-axis and return on the y-axis. These scatterplots are used to explain portfolio selection from Modern Portfolio Theory, and when analyzing past performance.It is common to measure risk and return for the past, and for the future time period using estimates. When creating the chart in Excel, select the XY Scatter chart type.")
			st.info("Correlations between holdings in a portfolio are of course a key component in financial risk management.Heat maps can be used for visualizing correlations among financial returns, and examine behaviour in both a stable and down market.")
			st.info("Profit after tax (PAT) or a gain after tax is essentially the amount of money that remains with the taxpayer after all the necessary deductions have been made. It is like a barometer that tells you how much profit a business has really earned.Capital expenditures (CapEx) are funds used by a company to acquire, upgrade, and maintain physical assets such as property, plants, buildings, technology, or equipment. CapEx is often used to undertake new projects or investments by a company.")
			st.info("Regression Analysis is a form of predictive analysis. We can use it to find the relation of a company’s performance to the industry performance or competitor business.The single (or simple) linear regression model expresses the relationship between the dependent variable (target) and one independent variable. Regression attempts to find the strength of that relationship.We use it to analyze the statistical relationship between sets of variables. Regression models usually show a regression equation representing the dependent variable as a function of the independent variable.")



	# if option=="Member's Home":
	# 	st.sidebar.write("")
		
	# 	choice=st.sidebar.radio('',['Login','SingUp'])

	# 	if choice == 'SingUp':
			
	# 		new_user = st.sidebar.text_input('Username')
	# 		new_passwd = st.sidebar.text_input('Password',type='password')
	# 		if st.sidebar.button('SignUp'):
	# 			create_usertable()
	# 			add_userdata(new_user,make_hashes(new_passwd))
	# 			st.sidebar.success("You have successfully created an account.Go to the Login Menu to login")
	# 	if choice=='Login':
	# 		user = st.sidebar.text_input('Username')
	# 		passwd = st.sidebar.text_input('Password',type='password')
	# 		if st.sidebar.checkbox('Login') :
	# 			create_usertable()
	# 			hashed_pswd = make_hashes(passwd)
	# 			result = login_user(user,check_hashes(passwd,hashed_pswd))
	# 			if result:
	# 				st.sidebar.success("Logged In as {}".format(user))
	# 				st.markdown(html_code.format("Member's Home"),unsafe_allow_html=True)
	# 				st.write("")
	# 				st.write("")
	# 				st.write("")
	# 				my_corner = st.selectbox('Select Option',('Add and Upload Notes', 'View All Notes & Documents', 'Upload Documents','Retrieve All Documents','Users'))

	# 				if my_corner=='Add and Upload Notes' and st.checkbox('Run'):

	# 					st.subheader("Add and Upload Notes")
	# 					create_table()
	# 					company_name = st.text_input("Enter Company Name",max_chars=50)
	# 					blog_title = st.text_input("Enter Title")
	# 					blog_article = st.text_area("Write your notes here",height=200)
	# 					blog_post_date = st.date_input("Date")
	# 					#uploaded_file = st.file_uploader("Upload Files",type=['xlsx','csv'])
	# 					if st.button("Add"):
	# 						add_data(company_name,blog_title,blog_article,blog_post_date)
	# 						st.success("Post:{} saved".format(blog_title))
	# 				if my_corner=='View All Notes & Documents'and st.checkbox('Run'):
	# 						st.subheader("View Articles")
	# 						all_titles = [i[0] for i in view_all_titles()]
	# 						postlist = st.selectbox("View Posts",all_titles)
	# 						post_result = get_blog_by_title(postlist)
	# 						for i in post_result:
	# 							b_author = i[0]
	# 							b_title = i[1]
	# 							b_article = i[2]
	# 							b_post_date = i[3]
	# 							st.text("Reading Time:{} mins".format(readingTime(b_article)))
	# 							st.markdown(head_message_temp.format(b_title,b_author,b_post_date),unsafe_allow_html=True)
	# 							st.markdown(full_message_temp.format(b_article),unsafe_allow_html=True)

	# 				if my_corner=='Upload Documents':
	# 					uploaded_file = st.file_uploader("Upload Files",type=['xlsx'])
	# 					df=pd.read_excel(uploaded_file)
	# 					st.dataframe(df)

	# 					if st.button('Upload'):


	# 						db_exist=True
	# 						i=0
	# 						while db_exist and i<=5:
	# 							try:
	# 								cann = sqlite3.connect(user+str(i)+'.db')
	# 								df.to_sql(user+str(i),cann)
									
	# 							except:
	# 								i+=1
	# 							else:
	# 								db_exist=False
	# 								cann.close()
	# 						st.success('Successfully Uploaded')

	# 				if my_corner=='Retrieve All Documents' and st.checkbox('Run'):
	# 					for i in range(0,6):
	# 						try:
	# 							cann = sqlite3.connect(user+str(i)+'.db')
	# 							df=pd.read_sql('SELECT * FROM '+user+str(i), cann)
	# 							st.dataframe(df)
	# 							st.markdown(overview_style.format(get_table_download_link(df)),unsafe_allow_html=True)
	# 							cann.close()
						
	# 						except:
	# 							break
	# 							cann.close()


						
									






					




















	# 				st.sidebar.write('')
	# 				st.sidebar.write('')
	# 			else:

	# 				st.sidebar.warning('Access Denied: Wrong Username/Password')
	# 				st.sidebar.write('')
	# 				st.sidebar.write('')

	# if st.sidebar.checkbox('SIP Calculator'):
	# 	P=st.sidebar.text_input("Principle Amount")
	# 	i = st.sidebar.slider('Select Annual Return %',0, 35)
	# 	n=st.sidebar.text_input("Investment period in years")
	# 	if st.sidebar.button('Calculate'):
	# 		P=float(P)
	# 		i=float((i/100)/12)
	# 		n=float(n)
	# 		n= n*12
	# 		FV =( P * (((1 + i)**(n)) - 1) / i)* (1 + i)
	# 		p_val=str(round(P*n,2))
	# 		f_val=str(round(FV,2))
	# 		st.sidebar.write('Total Amount Invested: '+p_val)
	# 		st.sidebar.write('Future Value: '+f_val)
	# 		if st.sidebar.button('Calculate'):
	# 			P=float(P)
	# 			i=float((i/100)/12)
	# 			n=float(n)
	# 			n= n*12
	# 			FV =( P * (((1 + i)**(n)) - 1) / i)* (1 + i)
	# 			p_val=str(round(P*n,2))
	# 			f_val=str(round(FV,2))
	# 			st.sidebar.write('Total Amount Invested: '+p_val)
	# 			st.sidebar.write('Future Value: '+f_val)
	# if st.sidebar.checkbox('CAGR Calculator',key='123'):
	# 	Pr=st.sidebar.text_input("Principle Amount",key='')
	# 	F=st.sidebar.text_input("Final Amount",key='')
	# 	t = st.sidebar.slider('Period',0, 35)
	# 	if st.sidebar.button('Calculate',key = "7239"):
	# 		Pr=float(Pr)
	# 		F=float(F)
	# 		cgr= 100*(((F/Pr)*(1/t))-1)
	# 		st.sidebar.write('Compounded Annual Growth: '+str(cgr)+'%')


	st.sidebar.write("")
	st.sidebar.write("")
	st.sidebar.write("")

	
				
				



	
			
					




			
				

			
				
			


if __name__=='__main__':
	main()
