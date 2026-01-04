import aiohttp


async def get_aiohttp_client() -> aiohttp.ClientSession:
    """Return a shared aiohttp client (placeholder)."""
    return aiohttp.ClientSession()
