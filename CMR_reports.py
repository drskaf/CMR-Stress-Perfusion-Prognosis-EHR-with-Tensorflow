from PyPDF2 import PdfFileReader
import pdfminer
temp = open('path to files.pdf', 'rb')
PDF_read = PdfFileReader(temp)
x = PDF_read.numPages
print(x)

pageobj = PDF_read.getPage(x+1)
text=PDF_read.extractText(0)

file1=open(r"C:\path to where to save files\\example.txt","a")
file1.writelines(text)
