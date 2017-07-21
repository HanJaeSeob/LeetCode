# -*- coding: utf-8 -*-
# reverse the sentence
#2017-7-18 by lee

def reverseSentence(s):

		#return ' '.join(s.split()[::-1]) 问题在于只是逆向输出，并没有reverse
		
		#l = s.split()
		#n = len(l)
		#for i in range(int(n/2)):
		 	#tmp = l[i]
			#l[i] = l[n-i-1]
		 	#l[n-i-1] = tmp
		 	#return ' '.join(l)
		 	#...以上是对的
		
		word = s.split()
		word.reverse()
		return ' '.join(word)
	 
if __name__ == '__main__':
	print reverseSentence("the sky is blue")