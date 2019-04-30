'''
	@author Qi Sun
	@desc Get some translated parallelism sentences
	@command example 
	python paraCrawler.py start_page end_page
'''

import sys
import requests

from bs4 import BeautifulSoup
from googletrans import Translator

def main():

	if len(sys.argv) != 3:
		print('Please enter page range you want to crawl')
		return

	if sys.argv[1].isdigit() == False \
		or sys.argv[2].isdigit() == False:
		print('Please enter numbers')
		return

	if int(sys.argv[1]) > int(sys.argv[2]):
		print('Please enter start page first')
		return

	# some settings before crawling
	start_page = int(sys.argv[1])
	end_page = int(sys.argv[2])
	url_prefix = 'http://www.ruiwen.com'
	url_middle = '/zuowen/paibiju/list_855_'
	headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3)\
				AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
	write_path = '../data/literary/webParallelism' +\
		str(start_page) + '-' + str(end_page) + '.txt'
	translator = Translator(service_urls=['translate.google.cn'])

	with open(write_path, 'w', encoding='utf-8') as writeF:
		for idx in range(start_page, end_page):
			index_html = requests.get(url_prefix+url_middle+str(idx)+'.html')
			content = index_html.text
			index_soup = BeautifulSoup(content, 'lxml')

			title_list = index_soup.find(class_='list_news')
			links = title_list.find_all("a")
			for link in links:
				final_url = url_prefix + link['href']
				final_html = requests.get(final_url)
				final_soup = BeautifulSoup(final_html.text, 'lxml')
				content_class = final_soup.find(class_='content')
				contents = content_class.find_all("p")
				for content in contents:
					text = content.text
					# if length of a sentence less than 15
					# we cant regard it as a parallelsim
					if len(text) < 15:
						continue
					if text[2].isdigit() == True:
						try:
							# because samples of parallelsim on websites have standard format
							# we choose to 
							en_text = translator.translate(text[2:], src='zh-cn', dest='en').text
							cn_text = translator.translate(en_text, src='en', dest='zh-cn').text
							print(cn_text)
							writeF.write(cn_text + '\n')
						except Exception:
							pass

if __name__ =="__main__":
	main()