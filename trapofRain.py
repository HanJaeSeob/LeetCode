# -*- coding: utf-8 -*-
def trap(height):
    '''
    #时间复杂度高
    count = 0
    for i in range(1,len(height)-1):
        Lmax = height[i]
        Rmax = height[i]
        for k in range(0,i):
            if height[k] > Lmax:
                Lmax = height[k]
        for k in range(i+1,len(height)):
            if height[k] > Rmax:
                Rmax = height[k]

        num = min(Lmax,Rmax) - height[i]
        count += num
        print Lmax,Rmax,num
    return count
    '''
    count = 0
    Lmax = [0 for i in range(len(height))]
    leftmax = 0
    for i in range(len(height)):
        if height[i] > leftmax:
            leftmax = height[i]
        Lmax[i] = leftmax
    rightmax = 0
    for i in reversed(range(len(height))):
        if height[i] > rightmax:
            rightmax = height[i]
        num = min(Lmax[i],rightmax) - height[i]
        count += num
    return count



if __name__ == '__main__':
    height = [0,1,0,2,1,0,1,3,2,1,2,1]
    print trap(height)
        
