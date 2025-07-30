import sys
sys.path.append(r'C:/IT/crawl4AI/crawl_lib')


from typing import List
import asyncio
from crawl4ai import *
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy

async def demo_deep_crawl():
    print("\n=== Deep Crawling ===")

    filter_chain = FilterChain([DomainFilter(allowed_domains=["ofac.treasury.gov"])])

    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=1,
        max_pages=1000,
        filter_chain=filter_chain
    )

    async with AsyncWebCrawler(config=BrowserConfig(
        viewport_height=800,
        viewport_width=1200,
        headless=True,
        verbose=True,
    )) as crawler:
        results: List[CrawlResult] = await crawler.arun(
            url='https://ofac.treasury.gov/sanctions-programs-and-country-information/russian-harmful-foreign-activities-sanctions',
            config=CrawlerRunConfig(deep_crawl_strategy=deep_crawl_strategy),
        )

        print(f"Deep crawl returned {len(results)} pages:")
        for i, result in enumerate(results):
            depth = result.metadata.get("depth", "unknown")
            print(f"{i+1}. {result.url} (Depth: {depth})")

if __name__ == "__main__":
    asyncio.run(demo_deep_crawl())
