class FrameCounter():
    # Use for counting elapsed frame in the live stream
    def __init__(self):
        self.frame = 0

    def __call__(self, count=0):
        self.frame += count
        return self.frame