import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(
                url="https://haznain.com",
            )
            # Assuming 'result.markdown' is the attribute you expect to hold the output
            with open("output.txt", "w", encoding="utf-8") as file:
                file.write(result.markdown)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
