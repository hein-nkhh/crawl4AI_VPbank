import sys
sys.path.append(r'C:/IT/crawl4AI/crawl_lib')


from typing import List
import asyncio
from crawl4ai import *
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy

async def demo_fit_markdown():
    print("\n=== 3. Fit MarkDown with LLM content Filter ===")

    markdown_gen = DefaultMarkdownGenerator(content_filter=PruningContentFilter())
    
    run_cfg = CrawlerRunConfig(
        markdown_generator=markdown_gen,
        stream=True,
        verbose=False
    )
    
    async with AsyncWebCrawler() as crawler:
        result: List[CrawlResult] = await crawler.arun(
            url = 'https://tuoitre.vn/ba-truong-my-lan-bi-tuyen-y-an-tu-hinh-20241203120659551.htm',
            config=run_cfg
        )

        print(f"Fit markdown: {result.markdown.fit_markdown}")
        
        
if __name__ == "__main__":
    asyncio.run(demo_fit_markdown())