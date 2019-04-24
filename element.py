import logging
import itertools

import numpy
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

class SchwellwertElement:
    
    def __init__(self, size, gewichte=None, schwellwert=None, lernrate=1):
        self.size = size
        self.gewichte = gewichte if gewichte is not None else numpy.random.randint(-5, 6, size)
        self.schwellwert = schwellwert if schwellwert is not None else numpy.random.randint(-5, 5)
        self.lernrate = lernrate

    def ausgabe(self, eingabe):
        summe = numpy.sum(self.gewichte * eingabe)
        return 1 if summe >= self.schwellwert else 0

    def plot(self):
        assert self.size == 2
        x = numpy.arange(-1, 3)
        y = numpy.polyval([-self.gewichte[0]/self.gewichte[1], self.schwellwert/self.gewichte[1]], x=x)
        plt.plot(x,y)

    def __repr__(self):
        return '<{} gewichte={} schwellwert={}>'.format(self.__class__.__name__, self.gewichte, self.schwellwert)


def show_element(element):
    LOGGER.debug('zeige %s', element)
    assert element.size == 2, "Funktioniert nur mit 2 Eingaben"
    for eingabe in itertools.product([0,1], [0,1]):
        eingabe = numpy.array(eingabe)
        ausgabe = element.ausgabe(eingabe)
        plt.plot(*eingabe,'ro', label='{}={}'.format(eingabe, ausgabe), color='black' if ausgabe else 'gray')
    element.plot()
    plt.legend()
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.title(element)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    LOGGER.debug('leaving')

def online_training(element, richtige_ausgaben, zeige_elmente=False):
    fehler_sum = 1
    epoche = 1
    LOGGER.info('starte online training mit: %s', element)
    if zeige_elmente: show_element(element)
    while(fehler_sum > 0):
        LOGGER.info('epoche: %s', epoche)
        fehler_sum = 0
        for eingabe, richtige_ausgabe in richtige_ausgaben:
            LOGGER.info('training mit: eingabe = %s richtige ausgabe = %s', eingabe, richtige_ausgabe)
            ausgabe = element.ausgabe(eingabe)
            fehler = richtige_ausgabe - ausgabe
            if ausgabe != richtige_ausgabe:
                LOGGER.info('element erzeugt falsche ausgabe = %s', ausgabe)
                schwellwert_neu = element.schwellwert - element.lernrate * fehler
                gewichte_neu = element.gewichte + element.lernrate * fehler * eingabe
                LOGGER.info('schwellwert_neu = %s', schwellwert_neu)
                LOGGER.info('gewichte_neu = %s', gewichte_neu)
                element.schwellwert = schwellwert_neu
                element.gewichte = gewichte_neu
                fehler_sum += abs(fehler)
            else: 
                LOGGER.info('element erzeugt richige ausgabe = %s', ausgabe)
            if zeige_elmente: show_element(element)
        epoche += 1
    LOGGER.info('element nach training: %s', element)

def batch_training(element, richtige_ausgaben, zeige_elmente=False):
    fehler_sum = 1
    epoche = 1
    LOGGER.info('starte batch training mit: %s', element)
    if zeige_elmente: show_element(element)
    while(fehler_sum > 0):
        LOGGER.info('epoche: %s', epoche)
        fehler_sum = 0
        schwellwert_sum = 0
        gewichte_sum = numpy.zeros(element.size, dtype='int')
        for eingabe, richtige_ausgabe in richtige_ausgaben:
            LOGGER.info('training mit: eingabe = %s richtige ausgabe = %s', eingabe, richtige_ausgabe)
            ausgabe = element.ausgabe(eingabe)
            fehler = richtige_ausgabe - ausgabe
            if ausgabe != richtige_ausgabe:
                LOGGER.info('element erzeugt falsche ausgabe = %s', ausgabe)
                schwellwert_sum -= element.lernrate * fehler
                gewichte_sum += element.lernrate * fehler * eingabe
                LOGGER.info('schwellwert_sum = %s', schwellwert_sum)
                LOGGER.info('gewichte_sum = %s', gewichte_sum)
                fehler_sum += abs(fehler)
            else: 
                LOGGER.info('element erzeugt richige ausgabe = %s', ausgabe)
        # Anpassung Schwellwert und Gewichte mit Summe
        element.schwellwert += schwellwert_sum
        element.gewichte += gewichte_sum
        if zeige_elmente: show_element(element)
        epoche += 1
    LOGGER.info('element nach training: %s', element)

def main():
    # element = SchwellwertElement(2, gewichte=numpy.array([0,0]), schwellwert=0)
    element = SchwellwertElement(3)
    # element = SchwellwertElement(2, gewichte=numpy.array([0,1]), schwellwert=1, lernrate=10)
    
    richtige_ausgaben = [
        (numpy.array([0,0,0]), 1),
        (numpy.array([0,0,1]), 1),
        (numpy.array([0,1,0]), 0),
        (numpy.array([0,1,1]), 1)
    ]
    online_training(element, richtige_ausgaben, zeige_elmente=False)
    #show_element(element)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()