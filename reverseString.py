# -*- coding: utf-8 -*-
# Reverse string
# 2017-7-18 by lee

def reverseString(string):
	#return ''.join(list(string)[::-1])
	'''
	l = list(string)
	n = len(l)
	for i in range(int(n/2)):
		tmp = l[i]
		l[i] = l[n-i-1]
	 	l[n-i-1] = tmp
	return ''.join(l)
	以上是正确的
	'''
	l = list(string)
	l.reverse()
	return ''.join(l)


if __name__ == '__main__':
	print reverseString("hello")