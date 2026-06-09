# Source - https://stackoverflow.com/a/66867816
# Posted by Tielessin, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-09, License - CC BY-SA 4.0

import asyncio
import math
import pygame


async def consume_input(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    try:
        while True:
            input = await queue.get()
            if input is None:
                break
            if input != {}: 
                print(input)

    except Exception as e:
        print(e)

async def amain(loop: asyncio.AbstractEventLoop):
    
    input_queue = asyncio.Queue(maxsize=1)

    input = asyncio.create_task(input_task(input_queue, loop))
    consume = asyncio.create_task(consume_input(input_queue, loop))
    
    result = await asyncio.gather(input, consume, return_exceptions=True)
    for r in result:
        if isinstance(r, Exception):
            print(f"[amain] failed - task raised {r}")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(amain(loop))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == '__main__':
    main()