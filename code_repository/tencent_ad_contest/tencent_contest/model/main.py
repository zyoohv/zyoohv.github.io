#! /usr/bin/python3
def main():
    try:
        while True:
            line1 = input().strip().split(' ')
            n = int(line1[0])
            name_list = []
            num_list = [0]
            for i in range(1, len(line1)):
                if i % 2 == 1:
                    name_list.append(line1[i])
                else:
                    num_list.append(int(line1[i]))

            ans = [0 for _ in range(len(num_list))]
            m = int(input())

            for i in range(len(num_list) - 1, 0, -1):
                ans[i] = m % num_list[i]
                m = int(m / num_list[i])
            ans[0] = m

            add = 0
            if ans[1] * 2 >= num_list[1]:
                add = 1

            print("{} {}".format(ans[0] + add, name_list[0]))

            add = 0
            if n > 2 and ans[2] * 2 >= num_list[2]:
                add = 1

            if ans[1] + add >= num_list[1]:
                print("{} {} {} {}".format(ans[0] + 1, name_list[0], 0,
                                           name_list[1]))
            else:
                print("{} {} {} {}".format(ans[0], name_list[0], ans[1] +
                                           add, name_list[1]))

    except EOFError:
        pass


if __name__ == '__main__':

    main()

