import threading
import time

def test_hreading(num:int) -> int:
    time.sleep(3)
    print(f" Result is {num*num}")
    
if __name__ == "__main__":
    lst = [2,3,4,5,6]
    new_lst = []
    for number in lst:
        timestamp = int(time.time())
        timestamp = threading.Thread(target=test_hreading, args=(number,))
        timestamp.start()
        new_lst.append(timestamp)
    for variable in new_lst:
        variable.join()
    print("Done")
