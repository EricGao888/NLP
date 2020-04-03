def compute_time(start, end):
    minutes = int((end - start) / 60)
    seconds = (end - start) % 60
    print("Total time cost: %d minutes %d seconds" % (minutes, seconds))