import asyncio
import httpx


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict:
    """Fetch a URL asynchronously and return JSON."""
    response = await client.get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()


async def fetch_multiple(urls: list[str]) -> list[dict]:
    """Fetch all URLs concurrently - much faster than sequential."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_json(client, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]


async def main():
    # JSONPlaceholder - a free public test API
    urls = [
        f"https://jsonplaceholder.typicode.com/posts/{i}" for i in range(1, 4)
    ]
    posts = await fetch_multiple(urls)
    for post in posts:
        print(f"Post {post['id']}: {post['title'][:40]}")


asyncio.run(main())