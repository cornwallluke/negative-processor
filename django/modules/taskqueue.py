import threading
import sys
import time
import queue
import logging

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

class myQ:
    _instance = None
    _queue = queue.Queue()
    _thread = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(myQ, cls).__new__(cls)
        return cls._instance

    @classmethod
    def addAction(cls, action):
        cls._queue.put(action)

    @classmethod
    def startQueue(cls):
        if cls._thread is None:
            cls._thread = threading.Thread(target=cls._mainloop, daemon=True)
            cls._thread.start()

    @classmethod
    def _mainloop(cls):
        while True:
            try:
                # if cls._queue.not_empty:
                cls._queue.get()()
                logging.debug("completed")
                time.sleep(.1)
            except:
                logging.debug("non function given to queue")
