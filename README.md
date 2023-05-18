# Oppsummeringer av e-poster med transformers
Denne koden er skrevet i sammenheng med bacheloroppgave våren 2023 hos NTNU.

Koden tilbyr følgende funksjonaliteter:
* Trene en sequence-to-sequence modell ved hjelp av huggingface med valgt modellutgangspunkt og datasett. Trening er foreløpig kun testet på T5 modeller, men skal fungere med `Seq2Seq` modeller fra huggingface.
* Evaluere modellen sin prestasjon på et datasett i form av ROUGE score.
# Installasjon
Koden er kun testet på linux med `python 3.9`. Pakkene prosjektet avhenger er: `numpy`, `torch`, `transformers`, `datasets`, `evaluate` og `spacy`.

Det er også nødvendig å installere `nb_core_news_sm` fra spacy ved følgende kommando: 

    python -m spacy download nb_core_news_sm

# Trening
`src/train.py` benyttes for å trene en modell. Kjør filen ved følgende kommando:

    python src/train.py -param1 <value> -param2 <value> ... -paramN <value>

Som antydet ved kjøringskommandoen er det flere mulige instillinger for trening:
| Param | Description | Default|
|--|--|--|
| model_checkpoint | Pre-trained modell som skal fine-tunes. Spesifiser IDen til modellen (fra huggingface) Eksempel: `north/t5_base_NCC`. *Dette feltet er obligatorisk*. |None|
|dataset_path|Datasettet som skal brukes for å trene, må være fra huggingface. Spesifiser IDen til datasettet, for eksempel: `cnn_dailymail`. Datasettet må være inndelt i `test`, `validation` og `train`. *Dette feltet er obligatorisk*.|None|
|model_save_name|Navnet til modellen som lagres etter trening er fullført. Lokalt blir modellen lagret med suffix -local og log lagres med suffix -log. På huggingface lagres modellen tilsvarende model_save_name. *Dette feltet er obligatorisk*.|None|
|max_input_length| Lengste input lengde under trening. |512|
|max_target_length|Lengste target (riktig output) under trening. |256|
|batch_size|Antall treningseksempler (par av input og riktig output) modellen prosesserer før modellparametere (vekter og _bias_) oppdateres under trening.|8|
|num_train_epochs|Antall epoker spesifiserer hvor mange ganger en modell går igjennom hele treningsdatasettet under trening.|8|
|lr|Learning rate|5.6e-5|
|weight_decay| Weight decay|0.01|
|input_name|Navn på kolonne i datasettet som tilsvarer input under trening.|text|
|label_name|Navn på kolonne i datasettet som tilsvarer target/label (riktig output) under trening.|label|
|gold_label_name|Navn på kolonne med gold labels i treningsdatasettet. Ignorer dette feltet dersom det kun er en label kolonne. |goldlabel|
|logging_level|Hvilket level av logging. Enten `debug`, `info`, `warning`, `error` eller `critical`.|debug|
|save_online|Dette feltet skal ha verdien `y` dersom modell skal lagres på huggingface etter trening er vellykket.|n|


Eksempel bruk: `python src/train.py -model_save_name t5_base_NCC-normail -model_checkpoint north/t5_base_NCC -dataset_path BiasedLiar/nor_email_sum`

# Evaluering
src/eval.py benyttes for å evaluere en modell.  Kjør filen ved følgende kommando: 

    python src/eval.py -param1 <value> -param2 <value> ... -paramN <value>
Som antydet ved kjøringskommandoen er det flere mulige instillinger for evaluering:
| Param | Description | Default|
|--|--|--|
| model_checkpoint | Modell som skal evalueres. Spesifiser IDen til modellen (fra huggingface) Eksempel: `north/t5_base_NCC`. *Dette feltet er obligatorisk*. |None|
|dataset_path|Datasettet som skal brukes for å evaluere, må være fra huggingface. Spesifiser IDen til datasettet, for eksempel: `cnn_dailymail`. Datasettet må ha inndelingen `test`, som brukes for evaluering. *Dette feltet er obligatorisk*.|None|
|max_summary_length| Maks lengde for oppsummeringene som genereres ved evaluering. |512|
|input_name|Navn på kolonne i datasettet som tilsvarer input under trening.|text|
|label_name|Navn på kolonne i datasettet som tilsvarer target/label (riktig input) under trening.|label|
|gold_label_name|Navn på kolonne med gold labels i treningsdatasettet. Ignorer dette feltet dersom det kun er en label kolonne. |goldlabel|
|logging_level|Hvilket level av logging. Enten `debug`, `info`, `warning`, `error` eller `critical`.|debug|


Eksempel bruk: `python src/eval.py -model_checkpoint BiasedLiar/t5_base_NCC-normail -dataset_path BiasedLiar/nor_email_sum`
