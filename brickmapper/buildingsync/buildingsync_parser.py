from lxml import etree
from pathlib import Path

file_etree = etree.parse(Path(__file__).parent  / "BuildingSync.sxd")
ns = file_etree.getroot().nsmap

class BuildingSyncParser:
    def _get_term_definitions(self):
        defns = []
        for element in file_etree.findall(".//xs:element", namespaces=ns):
            if element.get("ref") is None:
                name = element.get("name")
                documentation = element.find("./xs:annotation/xs:documentation", namespaces=ns)
                documentation =  "" if documentation is None else documentation.text

                defns.append({"name": name, "term_definition": documentation})
        
        return defns
                
