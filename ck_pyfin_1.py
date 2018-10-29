
def candy(arr):
    dp = [0]*(len(arr))
    n = len(arr)
    if(arr[n-1] <= arr[n-2]):
            dp[n-1] = 1
    i = n-2
    while i > 0:
        if(arr[i] <= arr[i-1]):
            dp[i] = dp[i] + dp[i+1] + 1
        i = i - 1
    dp[0] = 1 + dp[1]
    sum = dp[0]
    i = n-1
    for i in range(1, n-1):
        if(arr[i] >= arr[i-1] and arr[i] >= arr[i+1]):
            mod = dp[i+1] if(arr[i+1] > arr[i-1]) else dp[i-1]
            dp[i] = dp[i] + mod + 1
        elif(arr[i] >= arr[i-1]):
            dp[i] = dp[i] + dp[i-1] + 1
        sum = sum + dp[i]  
    if arr[n-1] > arr[n-2]:
        dp[n-1] = dp[n-2] + 1 
    for i in range(0, n-1):
        if(arr[i] == arr[i+1]):
            print(i, end = ' ')
    print()
#    print(dp)    
    return sum

if __name__ == '__main__': 
    in_file = open("in_1.txt", 'r')
    line = in_file.readline()
    arr = []
    while line:
        arr.append(line)
        line = in_file.readline()
    sum  = candy(arr)
    print(sum)
