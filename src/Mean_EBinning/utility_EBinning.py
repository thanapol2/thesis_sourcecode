

def count_segment_IOU(list_ts, ans_set):
    count = 0
    bin = 0
    max_iou = 0
    for index, ts in enumerate(list_ts[:-1]):
        bin_range = range(int(ts), int(list_ts[index + 1]))
        temp = len(ans_set.intersection(bin_range))
        if temp > 0:
            count = count + len(bin_range)
            bin = bin + 1
            iou = temp / (len(bin_range) + len(ans_set) - temp)
            if max_iou < iou:
                max_iou = iou
    return count, bin, max_iou