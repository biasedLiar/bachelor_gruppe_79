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
`src/train.py` benyttes for å trene en modell. Kjør filen ved følgende kommando:

    python src/train.py -param1 <value> -param2 <value> ... -paramN <value>

Som antydet ved kjøringskommandoen er det flere mulige instillinger for trening:
| Param | Description | Default|
|--|--|--|
| model_checkpoint | Pre-trained modell som skal fine-tunes. Spesifiser IDen til modellen (fra huggingface) Eksempel: `north/t5_base_NCC`. *Dette feltet er obligatorisk*. |None|
|dataset_path|Path til datasettet som skal brukes for å trene, må være fra huggingface. *Dette feltet er obligatorisk*.|None|
|model_save_name|Navnet til modellen som lagres etter trening er fullført. Lokalt blir modellen lagret med suffix -local og log lagres med suffix -log. På huggingface lagres modellen tilsvarende model_save_name. *Dette feltet er obligatorisk*.|None|
|dataset_type|Type datasett. Dersom det er et datasett fra huggingface skal verdien være `hf`. Ellers er det endelsen på den lokale filen som skal trenes med, for eksempel `csv`. NB! Datasettet må være delt inn i train, test og validation.|hf|
|max_input_length| Lengste input lengde mtp. på trening. |512|
|max_target_length|Lengste target (hva som trenes mot) lengde mtp. på trening. |256|
|batch_size||8|
|num_train_epochs||8|
|lr|Learning rate|5.6e-5|
|weight_decay||0.01|
|input_name|Navn på kolonne i datasettet som tilsvarer input under trening.|text|
|label_name|Navn på kolonne i datasettet som tilsvarer target/label under trening.|label|
|gold_label_name|Navn på kolonne med gold labels i treningsdatasettet. Ignorer dette feltet dersom det kun er en label kolonne. |goldlabel|
|logging_level|Hvilket level av logging. Enten `debug`, `info`, `warning`, `error` eller `critical`.|debug|
|save_online|Dette feltet skal ha verdien y dersom modell skal lagres på huggingface etter trening er vellykket.|n|

# Evaluering
src/eval.py benyttes for å evaluere en modell.  Kjør filen ved følgende kommando: 

    python src/eval.py -param1 <value> -param2 <value> ... -paramN <value>
Som antydet ved kjøringskommandoen er det flere mulige instillinger for evaluering:
| Param | Description | Default|
|--|--|--|
| model_checkpoint | Pre-trained modell som skal fine-tunes. Spesifiser IDen til modellen (fra huggingface) Eksempel: `north/t5_base_NCC`. *Dette feltet er obligatorisk*. |None|
|dataset_path|Path til datasettet som skal brukes for å trene, må være fra huggingface. *Dette feltet er obligatorisk*.|None|
|dataset_type|Type datasett. Dersom det er et datasett fra huggingface skal verdien være `hf`. Ellers er det endelsen på den lokale filen som skal trenes med, for eksempel `csv`. NB! Datasettet må være delt inn i train, test og validation.|hf|
|max_summary_length| Lengste input for oppsummeringer som genereres ved evaluering. |512|
|input_name|Navn på kolonne i datasettet som tilsvarer input under trening.|text|
|label_name|Navn på kolonne i datasettet som tilsvarer target/label under trening.|label|
|gold_label_name|Navn på kolonne med gold labels i treningsdatasettet. Ignorer dette feltet dersom det kun er en label kolonne. |goldlabel|
|logging_level|Hvilket level av logging. Enten `debug`, `info`, `warning`, `error` eller `critical`.|debug|
