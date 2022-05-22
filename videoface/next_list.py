from bisect import insort_right, bisect_left


class NextList:
    def __init__(self):
        self.list = []

    def __setitem__(self, key, value):  # , lower=0
        insort_right(self.list, (key, value), key=lambda f: f[0])

    def __getitem__(self, key):  # , lower=0
        i = bisect_left(self.list, key, key=lambda f: f[0])
        return self.list[i]

    def __iter__(self):
        for v in self.list:
            yield v

    def __len__(self):
        return len(self.list)

    def next_key(self, key):
        i = bisect_left(self.list, key, key=lambda f: f[0])
        if i > len(self.list)-1:
            return None

        if self.list[i][0] == key:
            if i == len(self.list)-1:
                return None

            return self.list[i+1][0]

        return self.list[i][0]

    def between(self, start, end):
        if start > end:
            start, end = end, start

        i = bisect_left(self.list, start, key=lambda f: f[0])
        if i == len(self.list):
            return False

        if self.list[i][0] < end:
            return True
        return False
