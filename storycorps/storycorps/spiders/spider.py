# -*- coding: utf-8 -*-

import re
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.http import Request
from storycorps.items import StorycorpsItem

class StorycorpsSpider(CrawlSpider):
	name = "StoryCorps"
	allowed_domain = ['storycorps.org']

	start_urls = ['https://storycorps.org/podcast',
		'https://storycorps.org/podcast/page/1/',
		'https://storycorps.org/podcast/page/2/',
		'https://storycorps.org/podcast/page/3/',
		'https://storycorps.org/podcast/page/4/',
		'https://storycorps.org/podcast/page/5/',
		'https://storycorps.org/podcast/page/6/']

#rules = (
#		Rule(LinkExtractor(allow=(r'.+storycorps.org/podcast/storycorps.+')), callback = "parse_item"),
#		Rule(LinkExtractor(deny=(r'.+storycorps.org/podcast')), callback = "parse_item"),
#	)

	link_extractor = {
		'page': SgmlLinkExtractor(allow = '/storycorps.*\d+.*\w+/$'),
	}

	def parse(self, response):
		self.log("hi, this is an item page! %s" % response.url)

		for link in self.link_extractor['page'].extract_links(response):
			yield Request(url = link.url, callback = self.parse_content)

	def parse_content(self, response):
		item = StorycorpsItem()
		item['name'] = response.xpath('//*[@id="gradient"]/div/div/@data-post-name').extract()[0].encode('utf-8')
		audio_source = response.xpath('//*[@id="gradient"]/div/div/section[5]/div[2]/audio/@src').extract()[0]
		item['audio'] = audio_source[:audio_source.find('?')].encode('utf-8')
		item['link'] = response.url

		all_script = response.xpath('//*[@class="modal-dialog"]/div[@class="modal-content"]/div[@class="modal-body"]/div[@class="transcript-container"]/p')
#self.log("all_script %s" % all_script)
		if (all_script == []):
			all_script = response.xpath('//main/article/section')

		full_script = []
		for sentence in all_script:
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
			line = line.strip('\n').rstrip('\t')
			line = line.replace("\n", "").replace("\t", "")
			if (line != ""):
				if (line != "\t"):
					if (line != "\n"):
						full_script.append(line)
#line = line + '\n'
#full_script.append(line)
#		self.log("aaa")

		item['script'] = full_script
#		self.log("script %s" % item['script'])

		return item

