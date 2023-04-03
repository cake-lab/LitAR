#!/usr/bin/env python3
import sys
import fire
import importlib
import multiprocessing

entries = {
    'serve': {'module': 'service', 'func': 'start_service'}
}


class Launcher:
    def __dir__(self):
        return list(entries.keys())

    def __getattr__(self, key):
        if key not in entries.keys():
            return None

        m = importlib.import_module(entries[key]['module'])
        return getattr(m, entries[key]['func'])

    def __call__(self):
        print('🚀 Launch via the following commands:', ', '.join(entries.keys()))


if __name__ == "__main__":
    # I don't like the way that Fire uses help screen instead of
    # just printing messages to std out. So, patch it:
    fire.core.Display = lambda lines, out: print(*lines, file=out)

    fire.Fire(Launcher)
