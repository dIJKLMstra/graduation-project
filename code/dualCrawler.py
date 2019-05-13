'''
	@author Qi Sun
	@desc Crawl some poems because poems contain the most dual sentences
	@command example 
	python dualCrawler.py start_page end_page
'''

import sys
import requests

from bs4 import BeautifulSoup

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
	url_prefix = 'https://www.gushiwen.org/shiwen/default.aspx?page='
	url_middle = '&type=0&id=0'
	headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3)\
				AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
	write_path = '../data/literary/webDual' +\
		str(start_page) + '-' + str(end_page) + '.txt'

	with open(write_path, 'w', encoding='utf-8') as writeF:
		for idx in range(start_page, end_page):
			index_html = requests.get(url_prefix+str(idx)+url_middle)
			content = index_html.text
			index_soup = BeautifulSoup(content, 'lxml')

			contents = index_soup.find_all(class_='contson')
			for content in contents:
				text = content.text
				writeF.write(text)

if __name__ =="__main__":
	main()