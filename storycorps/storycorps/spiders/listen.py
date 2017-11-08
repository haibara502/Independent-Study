# -*- coding: utf-8 -*-

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.http import Request
from storycorps.items import StorycorpsItem

class StorycorpsSpider(CrawlSpider):

	name = "StoryCorpsListen"
	allowed_domain = ['storycorps.org']

	start_urls = ['https://storycorps.org/listen', 
		'https://storycorps.org/listen/page/1/',
		'https://storycorps.org/listen/page/5/',
		'https://storycorps.org/listen/page/9/',
		'https://storycorps.org/listen/page/13/',
		'https://storycorps.org/listen/page/17/',
		'https://storycorps.org/listen/page/21/',
		'https://storycorps.org/listen/page/25/',
		'https://storycorps.org/listen/page/29/',
		'https://storycorps.org/listen/page/33/',
		'https://storycorps.org/listen/page/37/',
		'https://storycorps.org/listen/page/41/']

	link_extractor = {
		'pages': SgmlLinkExtractor(allow = '/listen/page/\d+/$'),
		'page': SgmlLinkExtractor(allow = '/listen/\w+[-\w+]*/$'),
	}

	def parse(self, response):
		self.log("hi, this is all pages!")

		for link in self.link_extractor['pages'].extract_links(response):
			yield Request(url = link.url, callback = self.parse_page)
	
	def parse_pages(self, response):
		for link in self.link_extractor['pages'].extract_links(response):
			self.log(link.url)
			yield Request(url = link.url, callback = self.parse_pages)

		for link in self.link_extractor['page'].extract_links(response):
			self.log("Go to one page %s" % link)
			yield Request(url = link.url, callback = self.parse_page)

	def parse_page(self, response):
		self.log("hi, this is an item page! %s" % response.url)
		self.log("response.url")

		for link in self.link_extractor['page'].extract_links(response):
			yield Request(url = link.url, callback = self.parse_content)

	def parse_content(self, response):
		self.log("hi, this is the parse_content page!")
		item = StorycorpsItem()
		self.log(response.xpath('//main/div[@class="row"]/div[@class="hero"]/div[@class="player landscape top-story"]/@data-post-name').extract())
		item['name'] = response.xpath('//main/div[@class="row"]/div[@class="hero"]/div[@class="player landscape top-story"]/@data-post-name').extract()[0].encode('utf-8')
		audio_source = response.xpath('//*[@id="gradient"]/div/div/section[5]/div[2]/audio/@src').extract()[0]
		self.log("audio_source %s" % audio_source)
		item['audio'] = audio_source[:audio_source.find('?')].encode('utf-8')
		item['link'] = response.url

		all_script = response.xpath('//*[@class="modal-dialog"]/div[@class="modal-content"]/div[@class="modal-body"]/div[@class="transcript-container"]/p')
#self.log("all_script %s" % all_script)
		if (all_script == []):
			all_script = response.xpath('//main/article/section')

		full_script = []
		for sentence in all_script:
#self.log("sentence %s" % sentence)
			speaker = "".join(sentence.xpath('b/text()').extract()).encode('utf-8')
			content = "".join(sentence.xpath('span/text()').extract()).encode('utf-8')
			another_content = "".join(sentence.xpath('text()').extract()).encode('utf-8')
			another_speaker = "".join(sentence.xpath('strong/text()').extract()).encode('utf-8')
			another_another_content = "".join(sentence.xpath('p/text()').extract()).encode('utf-8')
			another_another_speaker = "".join(sentence.xpath('b/i/text()').extract()).encode('utf-8')
			another_another_another_content = "".join(sentence.xpath('i/span/text()').extract()).encode('utf-8')

			block_quote = sentence.xpath('blockquote/p')
			if (block_quote != []):
				sub_lines = []
				for sub_sentence in block_quote:
					sub_content = "".join(sub_sentence.xpath('text()').extract()).encode('utf-8')
					sub_line.append(sub_content + '\n')
				full_script.append(sub_lines)


			line = ""
			if (speaker != ""):
				line = speaker + ':'
			if (another_speaker != ""):
				line = line + another_speaker
			if (another_another_speaker != ""):
				line = line + another_another_speaker

			line = line + content
			if (another_content != ""):
				line = line + another_content
			if (another_another_content != ""):
				line = line + another_another_content
			if (another_another_another_content != ""):
				line = line + another_another_another_content
			line = line + '\n'	
			full_script.append(line)
#		self.log("aaa")

		item['script'] = full_script
#		self.log("script %s" % item['script'])

		return item

