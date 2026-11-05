# Gabarit d'ordre de mission

Gabarit texte, rempli pour chaque appel et affiché en entier avant le go. État au 5 novembre 2026 : aucun appel n'a été passé avec.

```
ORDRE DE MISSION n° AAAA-MM-JJ-xx · nom : <nom court pour le go>

Objectif (une phrase) :
Interlocuteur : <nom du service ou de la personne>
Numéro à appeler : <numéro>   Source du numéro : <site officiel, mail du JJ/MM...>
Langue : français / anglais
Créneau d'appel : <heures ouvrées du service>
Ce que l'agent peut dire du propriétaire : <nom ; numéro de dossier ou de réservation si utile ; rien d'autre>

L'agent PEUT ACCEPTER :
- créneaux : <issus de l'agenda du propriétaire, par ordre de préférence>
- budget maximum : <montant TTC> (au-delà : refuser ou renvoyer vers le propriétaire)
- engagements permis : <ex. réserver sans arrhes ; déplacer un rendez-vous existant>

L'agent DOIT REFUSER OU RENVOYER VERS LE PROPRIÉTAIRE :
- tout paiement, toute coordonnée bancaire, tout code reçu par SMS ou par mail
- tout engagement payant, contrat, abonnement, signature
- toute question sur la santé, les finances ou la vie privée du propriétaire hors mission
- toute demande de « parler au titulaire » : <transférer vers le propriétaire / proposer un rappel>

Si personne ne répond : <rappeler plus tard / laisser ce message : « ... »>
Si on lui demande s'il est une IA : oui, toujours.
Durée maximale : <x> min (attente comprise)
Modèle : Claude Opus 5.5 / Claude Sonnet 5.5     Voix : <voix du propriétaire / voix du catalogue>
Compte rendu attendu : résultat, citation de la confirmation, date, prix, suites
```

Premier message, joué d'abord et redit à chaque nouvel interlocuteur humain :

```
Bonjour, je suis l'assistant IA de <propriétaire>, avec une voix de synthèse. J'appelle de sa part au sujet de <objectif>. Cet appel est enregistré.
```

Armement de l'appel : le propriétaire répond `go envoi <nom>`, seul dans le message. Un go, un appel, valable 30 minutes.
