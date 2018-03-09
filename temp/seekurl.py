import sys
import urllib2

if __name__ == '__main__':
		idx = 0
		for url in sys.stdin.readlines():
				url = url.strip()
				fd = urllib2.urlopen(url)
				data = fd.read()
				outfile = file('output/%s.html' % idx, 'w')
				outfile.write(data)
				print 'complete [%d] [%s] len=%d' % (idx, url, len(data))
				idx += 1

