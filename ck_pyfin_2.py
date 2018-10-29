
def candy(arr):
    n = len(arr)
    i = n-2
    dp = [0]*n
    if(dp[n-1] <= dp[n-2]):
        dp[n-1] = 1
    print("the lenght is ", n)
    while(i > 0):
        if(arr[i] <= arr[i-1]):
            dp[i] = 1 + dp[i+1]
        else:
            j = i
            while(j > 0):
                chk = True
                if(arr[j] < arr[j-1]):
                    break
                j = j - 1
            if chk:
                val = i - j + 1
                while(i >= j):
                    dp[i] = val
                    val = val - 1
                    i = i - 1
                chk = False
            i = i+1
        i = i-1
    sum = 0
    print(dp)
    i = n-1
    while(i > 0):
        if(dp[i] == dp[i-1]):
            dp[i-1] = dp[i-1] + 1
        i = i -1
    for i in range(0, n-1):
        if(dp[i] == dp[i+1]):
            print(i, end = ' ')
        sum = sum + dp[i]
    print()
    return sum
if __name__ == '__main__': 
    in_file = open("in_1.txt", 'r')
    line = in_file.readline()
    arr = []
    while line:
        arr.append(line)
        line = in_file.readline()
    barr = [2, 4, 2, 6, 1, 7, 8, 9, 2, 1]
#    barr = [1, 2, 2]
    sum  = candy(arr)
    print(sum)
