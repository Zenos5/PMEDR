import sys

# To make the start,end line working, put this line of code before importing ElementTree
sys.modules['_elementtree'] = None
import xml.etree.ElementTree as ET
class LineNumberingParser(ET.XMLParser):
    def _start(self, *args, **kwargs):
        # Here we assume the default XML parser which is expat
        # and copy its element position attributes into output Elements
        element = super(self.__class__, self)._start(*args, **kwargs)
        element.start_line_number = self.parser.CurrentLineNumber
        element.start_column_number = self.parser.CurrentColumnNumber
        element.start_byte_index = self.parser.CurrentByteIndex
        return element

    def _end(self, *args, **kwargs):
        element = super(self.__class__, self)._end(*args, **kwargs)
        element.end_line_number = self.parser.CurrentLineNumber
        element.end_column_number = self.parser.CurrentColumnNumber
        element.end_byte_index = self.parser.CurrentByteIndex
        return element

