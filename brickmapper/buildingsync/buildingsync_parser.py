from lxml import etree
from pathlib import Path

file_etree = etree.parse(Path(__file__).parent  / "BuildingSync.sxd")
ns = file_etree.getroot().nsmap

class BuildingSyncParser:
    def _get_term_definitions(self):
        defns = []
        for element in file_etree.findall(".//xs:element", namespaces=ns):
            if element.get("ref") is not None:
                continue

            name = element.get("name")
            documentation = element.find("./xs:annotation/xs:documentation", namespaces=ns)
            documentation =  "" if documentation is None else documentation.text
            tree = file_etree.getpath(element)
            enumerations = element.findall("./xs:simpleType/xs:restriction/xs:enumeration", namespaces=ns)
                
            if enumerations:
                for enum in enumerations:
                    defns.append({
                        "name": name + "#" +  enum.get("value"), 
                        "definition": documentation, 
                        "tree": file_etree.getpath(enum)
                    })
            else:
                defns.append({
                    "name": name, 
                    "definition": documentation, 
                    "tree": tree
                })
        
        return defns
                
