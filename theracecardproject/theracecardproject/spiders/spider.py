# -*- coding: utf-8 -*-

import re
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.http import Request
from theracecardproject.items import TheracecardprojectItem

class TheracecardprojectSpider(CrawlSpider):
	name = "Theracecardproject"
	allowed_domain = ['theracecardproject.com']

	start_urls = ['http://theracecardproject.com']

	link_extractor = {
		'page': SgmlLinkExtractor(allow = '\.*theracecardproject.com/\w+[-\w+]*/$', deny = 'theracecardproject.com/the-race-card-project-wall/'),
	}

	def parse(self, response):
		self.log("hi, this is an item page! %s" % response.url)

		for link in self.link_extractor['page'].extract_links(response):
			yield Request(url = link.url, callback = self.parse_content)

	def parse_content(self, response):
		item = TheracecardprojectItem()
		item['author'] = response.xpath('//div[@class="entry-content"]/p/text()[1]').extract()[0]
		item['title'] = response.xpath('//head/title').extract()[0]
		item['link'] = response.url
		item['name'] = response.xpath('//head/link[@rel="canonical"]/@href').extract()[0]
		item['name'] = item['name'][29:]
		item['tag'] = ""
		tag = response.xpath('//div[@class="entry-content"]/p/text()[3]')
		if tag != []:
			item['tag'] = tag.extract()[0]
		item['location'] = response.xpath('//div[@class="entry-content"]/p/text()[2]').extract()[0]
		item['content'] = []
		content = response.xpath('//div[@class="entry-content"]/p')
		for line in content[1:]:
			texts = line.xpath('text()')
			item['content'].append(texts.extract())
		item['comments'] = []

		all_comment = response.xpath('//div[@class="post-body"]/div/div/div/div/div/p')
		self.log("link: %s all_comment : %s" % (item['link'], all_comment))
		for person in all_comment:
			person_name = person.xpath('//div[@class="post-body"]/header[@class="comment_header"]/span[@class="post-byline"]/span/span/a/text()').extract()[0]
			person_content = person.xpath('//div[@class="post-body-inner"]/div[@class="post-message-container"]/div[@class="publisher-anchor-color"]/div[@class="post-message"]/div/p/text()').extract()[0]
			one_comment = [person_name, person_content]
			item['comments'].append(one_comment)

		return item

