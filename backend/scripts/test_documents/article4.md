# Attribut (Programmierung)

Ein Attribut (von lateinisch attribuere ‚zuteilen‘, ‚zuordnen‘), auch Eigenschaft genannt, gilt im Allgemeinen als Merkmal, Kennzeichen, Informationsdetail etc., das einem konkreten Objekt zugeordnet ist. Dabei wird unterschieden zwischen der Bedeutung (z. B. Augenfarbe) und der konkreten Ausprägung (z. B. blau) des Attributs.

In der Informatik wird unter Attribut die Definitionsebene für diese Merkmale verstanden. Als solche werden sie analytisch ermittelt, definiert und beschrieben sowie für einen bestimmten Objekttyp als Elemente seiner Struktur festgelegt („modelliert“). Daten über die Objekte werden in dieser Struktur und nur mit ihrem Inhalt, den Attributwerten, gespeichert. Jedes Objekt repräsentiert sich somit durch die Gesamtheit seiner Attributwerte.

Jedem Attribut sind Regeln zugeordnet, die als Operationen bezeichnet werden. Daraus folgt, dass eine Objektdefinition durch die Definition von Datentypen erweitert werden kann. Ein Darstellungsformat, ein Standardwert sowie gültige Operationen und Einschränkungen (z. B. ist Division durch null nicht erlaubt) können an der Definition von Attributen beteiligt sein oder umgekehrt als Attribut des Objekttyps bezeichnet werden.

In der Computergrafik zum Beispiel können Linienobjekte beispielsweise Attribute wie Anfangspunkt und Endpunkt (mit Koordinaten als Werten), Breite (mit einer Gleitkommazahl als Wert), Farbe (mit beschreibenden Werten wie Rot, Gelb, Grün oder Blau oder in einem bestimmten Farbmodell definierte Werte wie im RGB-Farbraum) usw. aufweisen und Kreisobjekte können zusätzlich mit den Attributen Mittelpunkt und Radius definiert werden.

In Land- bzw. Geoinformationssystemen (GIS) ist die Datengrundlage von Objekten ihre Lage in einem horizontalen Koordinatensystem (meist ebene Gauß-Krüger-Koordinaten oder geografische Breite plus Länge). Alle weiteren Eigenschaften des Objekts (z. B. Höhe, Größe, Zweck, Zeitpunkt der Erfassung) werden den Lagekoordinaten als Attribut zugeordnet. Hingegen wird die Meereshöhe meist in ein zweidimensionales Modell (2½-D) realisiert, also als Objekt mit größerer Bedeutung; bei 3D-Modellen ist sie hingegen den Lagekorrdinaten gleichwertig. Objektattribute sind im Regelfall mit anderen Daten des GIS verknüpfbar, heute meist auch mit anderen, verwandten Datenbanken.

Verarbeitung
Zur Verarbeitung der Daten können Attribute und Attributwerte mengen-einschränkend benutzt werden:

zur Selektion: Auswahl einer Objekt-Teilmenge über ihre Attributwerte; Bsp.: Geburtsdatum < 1.1.2000
zur Projektion: Für die selektierten Objekte sollen nur bestimmte Attribute gelesen / verarbeitet werden; Bsp.: nur Name, Vorname, Geburtsdatum
C#
In der Programmiersprache C# sind Attribute Metadaten, die an ein Feld oder einen Codeblock wie Assemblys, öffentliche Variablen und Datentypen angehängt sind, und entsprechen Annotations in Java. Attribute sind sowohl für den Compiler als auch programmgesteuert durch Reflexion zugänglich. Mit Zugriffsmodifikator wie abstract, sealed oder public ist es möglich, Attribute zu erweitern.

Ihre spezifische Verwendung als Metadaten bleibt dem Entwickler überlassen und kann eine Vielzahl von Arten von Informationen zu bestimmten Anwendungen, Klassen und öffentlichen Variablen abdecken, die nicht instanzspezifisch sind. Die Entscheidung, ein bestimmtes Attribut als Eigenschaft verfügbar zu machen, bleibt ebenso dem Entwickler überlassen wie die Entscheidung, sie als Teil eines größeren Anwendungsframeworks zu verwenden.

Attribute werden als Klassen implementiert, die von System.Attribute abgeleitet sind. Sie werden häufig von den CLR-Diensten verwendet, z. B. COM-Interoperabilität, Remote Procedure Calls, Serialisierung und können zur Laufzeit abgefragt werden.

Positionsparameter wie der erste Parameter der obigen Typzeichenfolge sind Parameter des Konstruktors des Attributs. Namensparameter wie der Boolesche Parameter im Beispiel sind eine Eigenschaft des Attributs und sollten ein konstanter Wert sein.

Attribute sollten der XML-Dokumentation gegenübergestellt werden, die auch Metadaten definiert, jedoch nicht in der kompilierten Assembly enthalten ist und daher nicht programmgesteuert aufgerufen werden kann.[1]

Beispiele
Das folgende Beispiel in der Programmiersprache C# zeigt die Klassen Partei, Abgeordneter und Parlament, die öffentliche Attribute deklarieren. Die meisten dieser Attribute können von anderen Objekten gelesen, aber nicht geändert werden, weil die set-Methode mit dem Zugriffsmodifikator private deklariert ist. Das Attribut mitgliedsbeitrag der Klasse Partei und das Attribut maximalGroesse der Klasse Parlament können auch von anderen Objekten geändert werden.

Die Datentypen von Attributen können elementare Datentypen oder Klassen, also Objekttypen sein. Die meisten Attribute im Beispiel haben elementare Datentypen. Das Attribut vorsitzender der Klasse Abgeordneter hat den Objekttyp Abgeordneter. Das Attribut mitglieder der Klasse Partei hat den generischen Typ List<Person>, ist also eine Liste mit dem Typparameter Person.

class Person
{
public string vorname { get; private set; }
public string nachname { get; private set; }
public Date geburtsdatum { get; private set; }
public List<string> nationalitäten { get; private set; }
public string MailAdresse { get; private set; }
public string Postanschrift { get; private set; }
}

class Partei
{
public List<Person> mitglieder { get; private set; }
public double mitgliedsbeitrag { get; set; }
}

class Abgeordneter : Person
{
public Partei partei { get; private set; }
}

class Parlament
{
public List<Abgeordneter> abgeordnete { get; private set; }
public Abgeordneter vorsitzender { get; private set; }
public int maximalGroesse { get; set; }
}
Siehe auch
Attribut (UML)
Einzelnachweise
Hanspeter Mössenböck, University of Linz: C# Tutorial
Kategorie: Objektorientierte Programmierung
