import pstats

def main():

    p = pstats.Stats("bottleneck.txt")
    p.sort_stats("time").print_stats(100)


if __name__ == '__main__':
    main()
