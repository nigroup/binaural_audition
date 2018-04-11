def create_generator(dloader):
    while True:
        b_x, b_y = dloader.next_batch()
        if b_x is None or b_y is None:
            return
        yield b_x, b_y
