# SuperFastPython.com
# example of one producer and multiple consumers with threads
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd


def test_webcam():
    import cv2
    cap = cv2.VideoCapture()
    # The device number might be 0 or 1 depending on the device and the webcam
    cap.open(0, cv2.CAP_DSHOW)
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# producer task
def producer(queue_raw_data):
    for i in range(100):
        value = np.random.normal(size=(200, 100, 100))
        queue_raw_data.put([i, value])
    queue_raw_data.put(None)


# consumer task
def consumer(queue_raw_data, queue_processed_data, identifier):
    while True:
        item = queue_raw_data.get()
        if item is None:
            queue_raw_data.put(item)
            break
        result = [[np.mean(x), np.std(x)] for x in item[1]]
        result = np.append(item[0], np.reshape(result, (-1)))
        queue_processed_data.put(result)


def get_queue(q, pars=None):
    result = pd.DataFrame([q.get() for _ in range(q.qsize())])  # .set_index(0)
    if pars is not None:
        result.columns = ['frame'] + list(
            np.reshape([[f'{i}: {p}' for p in pars] for i in range(len(result.columns) // len(pars))], -1))
        result.set_index('frame', inplace=True, drop=True)
    return result


if __name__ == '__main__':
    queue_raw_data = Queue()
    queue_processed_data = Queue()

    consumers = [Thread(target=consumer, args=(queue_raw_data, queue_processed_data, i)) for i in range(3)]
    for consumer in consumers:
        consumer.start()

    producer = Thread(target=producer, args=(queue_raw_data,))
    producer.start()

    # wait for all threads to finish
    producer.join()
    for consumer in consumers:
        consumer.join()

    results = get_queue(queue_processed_data, pars=['mean', 'std'])
    print(results)
