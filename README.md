# Oppsummeringer av e-poster med transformers
Denne koden er skrevet i sammenheng med bacheloroppgave våren 2023 hos NTNU. Oppgaven er tilgjengelig fra lenken: LENKE.

Koden følgende funksjonaliteter:
* Trene en T5 modell ved hjelp av huggingface med valgt utgangspunkt og datasett
* Evaluere modellen sin prestasjon på et datasett i form av ROUGE score
# Installasjon
Koden er kun testet på linux med `python 3.9`. Pakkene prosjektet avhenger er: `numpy`, `datasets`, `spacy`, `transformers`, `torch`.

Det er også nødvendig å installere `nb_core_news_sm` fra spacy ved følgende kommando: 

    python -m spacy download nb_core_news_sm

Komplette versjoner av alle avhengigheter til prosjektet er gitt i filen `requirements.txt`  TODO: SKAL VI HA MED DETTE?

# Trening
src/train.py benyttes for å trene en modell. Kjør filen ved følgende kommando:

    python src/train.py -param1 <value> -param2 <value> ... -paramN <value>

Som antydet ved "param" er det flere instillinger for kjøring trening:
| Param | Value(s) | Default| Required |
|--|--|--|--|
| model_checkpoint |  |||



# Evaluering

