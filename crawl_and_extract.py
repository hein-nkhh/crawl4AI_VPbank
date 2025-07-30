import sys
sys.path.append(r'C:/IT/crawl4AI/crawl_lib')

from typing import List
import asyncio
from crawl4ai import *
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse

async def crawl_and_extract():

    print("\n=== Deep Crawling")
    
    filter_chain = FilterChain([DomainFilter(allowed_domains=["ofac.treasury.gov"])])

    # Deep Crawling
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth = 1, max_pages = 1, filter_chain = filter_chain
    )

    # Fit markdown
    markdown_gen = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter()
    )
    
    run_cfg = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_strategy,
        markdown_generator=markdown_gen,
    )
    
    async with AsyncWebCrawler(config=BrowserConfig(
        viewport_height=800,
        viewport_width=1200,
        headless=True,
        verbose=True,
    )) as crawler:
        results: List[CrawlResult] = await crawler.arun(
            url='https://ofac.treasury.gov/media/933066/download?inline',
            config=run_cfg
        )

        print(f"Deep crawl returned {len(results)} pages:")
        for i, result in enumerate(results):
            depth = result.metadata.get("depth", "unknown")
            print(f"{i+1}. {result.url} (Depth: {depth})")
                            
if __name__ == "__main__":
    # asyncio.run(crawl_and_extract())
    asyncio.run(crawl_and_extract())
