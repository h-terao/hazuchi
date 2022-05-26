def cycle(dataset):
    while True:
        for batch in dataset:
            yield batch
