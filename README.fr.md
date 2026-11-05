<picture><source media="(prefers-color-scheme: dark)" srcset="banner.svg"><img src="banner-light.svg" alt="Local voice agent" width="100%"></picture>

[English](README.md) · [Français](README.fr.md)

# Local voice agent

Ismaël Joffroy Chandoutis est artiste et cinéaste ; il fait tourner Claude Code comme couche d'exploitation de son atelier. En novembre 2026, il a fait monter un assistant vocal entièrement local sur une seule RTX 5090 : l'assistant répondait dans un navigateur, en français et en anglais, prenait des messages pour lui et rédigeait une fiche à la fin de chaque conversation. Le 5 novembre 2026, il lui a parlé lui-même pendant 57 secondes. Les transcriptions étaient justes, aucune phrase n'a été coupée, et les réponses arrivaient en 1,0 à 1,4 s côté serveur. Le rôle était mal choisi : l'assistant était réglé pour prendre les messages de tiers et n'avait rien à faire pour son propriétaire. Il voulait un agent qui téléphone à des services à sa place pour réserver, déplacer un rendez-vous, demander un prix ou relancer un dossier. Le pipeline local a été mis en pause en l'état, et l'agent qui appelle a été conçu ensuite, sur ElevenLabs avec Claude comme modèle de raisonnement, derrière un ordre de mission écrit et un go humain.

<img src="monde.jpg" alt="image-monde" width="100%">

Ce dépôt contient les deux parties. `pipeline/` est le pipeline local, avec le code qui a tourné et les mesures qu'il a produites. `caller/` documente l'agent qui appelle : son architecture, son gabarit d'ordre de mission et sa barrière d'envoi. Aucun code de l'agent qui appelle n'a encore tourné, donc aucun n'est publié.

## Partie 1 : un pipeline vocal sur une seule carte

### Architecture

```
navigateur (WebRTC) ──> bot Pipecat (bot.py, port 7860)
                          │
                          ├─ Silero VAD, 0,2 s de silence
                          ├─ Smart Turn v3 + garde à deux vitesses (fin de tour)
                          ├─ Kyutai stt-1b-en_fr en streaming (stt_server.py, port 8792)
                          ├─ Qwen3.6 35B-A3B, GGUF Q4_K_M, sans réflexion (llama.cpp, port 8080)
                          └─ Qwen3-TTS 0.6B CustomVoice servi par vLLM-Omni (port 8093)
                          │
                          └─ fin d'appel : le LLM local remplit une fiche (JSON + Markdown) dans calls/
```

Chaque brique écoute sur 127.0.0.1 et aucune requête ne quitte la machine. Dans le montage mesuré, llama.cpp tournait directement sous Windows et les autres briques dans WSL, sur la même machine. Le TTS utilisait une voix par défaut du modèle (`vivian` en français, `ryan` en anglais au banc). Aucune voix n'a été clonée.

Trois règles viennent des mesures :

1. Une annonce fixe est jouée à chaque connexion et dit que l'assistant est une IA. Avec la persona seule, 11 réponses sur 15 le disaient au banc. Les quatre autres : la question technique en anglais (3 passages sur 3) et un passage d'une question en français.
2. La fin de tour a deux vitesses. Si Smart Turn donne une probabilité d'au moins 0,985, le tour se termine tout de suite. En dessous, 0,25 s de silence supplémentaire sont exigées, et toute reprise de parole pendant l'attente annule la fin de tour. Les pauses à l'intérieur des questions de test duraient 0,28 à 0,36 s, et Smart Turn en a jugé trois finies, avec des probabilités de 0,855, 0,965 et 0,977. Les vraies fins de tour donnaient 0,938 à 0,991. Le seuil vient de cinq questions synthétiques seulement et se règle sur de vraies voix (`FAST_THRESHOLD`, `GUARD_SILENCE_S`).
3. 20 des 40 couches d'experts MoE du LLM restent en RAM (`--n-cpu-moe 20`). Le modèle Q4 entier demande environ 22,5 Go sur la carte, et il restait environ 14,3 Go une fois le STT et le TTS chargés.

La persona (`pipeline/persona.md`) a deux modes. En mode appel, l'assistant dit qu'il est une IA, ne prend aucun engagement, recueille le nom, le motif, le numéro et le message de l'appelant, puis les relit. En mode dictée, déclenché quand le propriétaire se nomme, il répond en quelques mots et range ce qui est dit en notes et en tâches. Les consignes sont rédigées en français, parce que le banc visait des appelants francophones et anglophones.

### Mesures (5 novembre 2026)

Tous les chiffres ont été mesurés sur la même machine le 5 novembre 2026. Les relevés bruts sont dans `pipeline/measurements/`.

Banc, 15 échanges (5 questions synthétiques, 3 passages chacune), latence de la fin de la question au premier son de la réponse, côté serveur :

| Variante | Coupures en pleine phrase | Latence médiane | Plage |
|---|---|---|---|
| Garde à deux vitesses | 0 sur 15 | 0,88 s | 0,57 à 1,17 s |
| Sans garde | 9 sur 15 | environ 0,65 s (14 passages, hors premier appel à froid) | |

Médiane par étape, avec la garde :

| Étape | Médiane |
|---|---|
| Silero voit 0,2 s de silence | 182 ms |
| Smart Turn v3 | 32 ms |
| Attente de la garde, quand elle joue (9 tours sur 15) | 289 ms |
| Vidage du STT | 126 ms |
| LLM jusqu'à la première phrase | 215 ms |
| dont premier token | 54 ms |
| Premier son du TTS | 32 ms |

Vrai bot Pipecat, sessions WebRTC complètes par une adresse HTTPS, avec `test_webrtc.py` qui joue un WAV à la place du micro :

| Mesure | Plage | Médiane |
|---|---|---|
| Côté serveur, de la fin de parole à « Bot started speaking » (11 tours) | 0,76 à 1,44 s | 1,13 s |
| Côté client, de la fin de la question au premier son reçu (12 tours) | 1,85 à 2,84 s | 2,2 s |

Le bot ajoute environ 0,25 s au banc : il attend une phrase entière avant le TTS, et le premier tour d'une session a un premier token plus lent (0,5 à 0,9 s, contre 0,2 s ensuite). L'écart d'environ 1 s entre serveur et client mêle le réseau, l'encodage Opus et le tampon de réception du client de test ; les mesures ne séparent pas ces parts. Le navigateur d'un téléphone a ses propres tampons, qui n'ont pas été mesurés.

Mémoire de la carte (`nvidia-smi`, en Mio) : 9 047 avant chargement (autres travaux de la machine et affichage), 12 258 avec le STT, 18 305 avec le STT et le TTS, 30 719 avec le LLM (12,4 Go pour le LLM avec 20 couches d'experts en RAM), 30 843 au pic du banc, sur une carte de 32,6 Go. Le débit du LLM est passé de 49 à 61 tokens par seconde avec 30 couches en RAM à 63 à 81 avec 20.

### Le relancer

Il faut une carte CUDA de 32 Go, Linux ou WSL, une version de llama.cpp avec `llama-server`, et le fichier GGUF `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. Les modèles `kyutai/stt-1b-en_fr` et `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` se téléchargent depuis Hugging Face au premier démarrage. Les versions de paquets installées sont listées dans `pipeline/requirements-*.txt` ; le montage mesuré utilisait trois environnements Python distincts (STT, TTS, bot).

```
cp .env.example .env              # renseigner MODEL_GGUF et l'interpréteur de chaque environnement
bash pipeline/scripts/start-all.sh
bash pipeline/scripts/stop-all.sh # arrête tout et affiche la mémoire de carte encore occupée
```

Placer un reverse proxy HTTPS devant le port 7860 : un navigateur ne donne le micro qu'à une page sécurisée. Si le proxy ajoute un en-tête avec l'utilisateur authentifié, `ALLOWED_USER_HEADER` et `ALLOWED_USER` font refuser par un 403 toute requête où cet en-tête manque ou nomme un autre utilisateur. Le proxy doit supprimer toute copie de l'en-tête envoyée par le client.

Pour refaire le banc, fabriquer une fois les questions (`python pipeline/make_questions.py`, Kokoro sur CPU), puis `python pipeline/bench.py --out resultats --tours 3`. Pour essayer le vrai bot sans micro : `BOT_URL=https://votre-hote python pipeline/test_webrtc.py questions/fr3.wav reponse.wav`. `make_scenarios.py` fabrique deux scénarios à plusieurs tours, un appelant qui laisse un message et une dictée. Les noms et le numéro des questions de test sont fictifs. Rien ne redémarre seul après un redémarrage de la machine.

### Limites

- Le banc a utilisé des voix synthétiques. Le seuil de la garde et la latence ressentie sur un téléphone restent à vérifier avec de vraies voix.
- Chaque échange du banc ne compte qu'un tour. La relecture du nom, du motif et du numéro sur plusieurs tours n'a été essayée que dans deux sessions simulées, où l'assistant n'a pas relu le numéro chiffre par chiffre.
- Le STT a mal entendu le nom de société du scénario de test, et la fiche a repris l'erreur. « Ismaël » était parfois compris « Ismail » en anglais.
- La dernière phrase de l'assistant, coupée par la déconnexion, n'est que partiellement dans la transcription.
- Un modèle local de cette taille répond et prend des notes. Sa capacité à tenir une négociation ou une décision soumise à des règles n'a pas été testée ; c'est la raison pour laquelle l'agent qui appelle s'appuie sur Claude.
- Aucune ligne téléphonique n'est branchée. Les appels arrivent d'un navigateur, en WebRTC.

## Partie 2 : un agent qui appelle, derrière une barrière d'envoi

État au 5 novembre 2026 : au stade de la conception. Aucun appel n'a été passé et aucun agent n'a été configuré pour cela. Cette partie décrit ce que la conception arrête, et le gabarit de `caller/` est un document texte.

### Architecture

```
propriétaire ── ordre de mission rédigé avec Claude Code ── « go envoi <nom> »
                          │
                          ▼
   API d'appel sortant ElevenLabs + variables de la mission
                          │
   téléphonie (numéro dédié à l'assistant, ou SIP) ── appel ── le service
                          │
   agent ElevenLabs : STT ElevenLabs → Claude → voix
       outils : agenda (lire les disponibilités, créer un événement), touches du clavier,
                détection de messagerie, transfert vers le propriétaire, signalement, fin d'appel
                          │
   fin d'appel → webhook : transcription + champs extraits
                          │
   compte rendu (Claude Code) : résultat, citation exacte de la confirmation, suites
```

Choix de conception consignés dans les notes sources :

- Un seul agent générique. Chaque appel reçoit sa mission par des variables dynamiques et des surcharges propres à l'appel (consigne, premier message, liste stricte des outils autorisés) : rien ne passe d'un appel à l'autre.
- Un modèle texte derrière la voix. Claude Opus 5.5 pour les missions qui demandent de décider, Claude Sonnet 5.5 pour les missions simples, chacun en secours de l'autre, avec un effort de raisonnement bas pendant la conversation. L'étude τ-Voice (Sierra et Princeton, mars 2026) donne 31 à 51 % de réussite sur des tâches téléphoniques soumises à des règles pour les agents « parole à parole », contre 85 % pour un modèle texte qui raisonne. La latence d'Opus 5.5 au téléphone n'a pas été mesurée.
- Agenda : lecture des disponibilités et création d'événements seulement. Les outils de modification et de suppression, qu'ElevenLabs marque lui-même comme privilégiés, restent désactivés.
- Après l'appel : collecte structurée (résultat, date obtenue, prix annoncé, citation exacte de la confirmation, suites) et transcription, pour que le compte rendu se vérifie sur l'appel.

### La barrière d'envoi

Avant l'appel :

1. Le propriétaire dit ce qu'il veut en une phrase.
2. Claude rédige l'ordre de mission d'après le gabarit, vérifie le numéro à sa source (site officiel, mail antérieur) et lit l'agenda pour les créneaux. Le numéro à appeler vient toujours de l'ordre de mission, jamais de l'agent.
3. L'ordre de mission est affiché en entier. Toute modification entraîne un nouvel affichage.
4. Le propriétaire répond `go envoi <nom de la mission>`, seul dans le message. Un go glissé dans un message qui porte d'autres consignes n'arme rien. Un go vaut pour un appel, dans les 30 minutes.

Pendant l'appel :

5. Le premier message dit, à chaque fois, qu'il s'agit de l'assistant IA du propriétaire avec une voix de synthèse, pourquoi il appelle, et que l'appel est enregistré. Il est redit à chaque nouvel interlocuteur humain, après un serveur vocal ou un transfert.
6. Ce que l'ordre de mission n'autorise pas reçoit « je dois vérifier avec lui, il vous recontacte », ou un transfert vers le propriétaire si l'ordre le prévoit.
7. Avant de raccrocher, l'agent récapitule à voix haute et demande confirmation.

Après l'appel :

8. Un compte rendu sous cinq minutes : résultat (obtenu, partiel, échec), ce qui a été convenu mot pour mot, prix, date, prochaines actions, écarts par rapport à l'ordre, liens vers l'enregistrement et la transcription.
9. Un rendez-vous confirmé devient tout de suite un événement d'agenda, avec le contexte en notes.
10. Un rappel ou une relance passe par un nouvel ordre de mission et un nouveau go.

Règles dures dans chaque mission : aucun paiement, aucune coordonnée bancaire, aucun code reçu par SMS ou par mail, aucun mot de passe, aucun contrat ni abonnement, jamais se présenter comme le propriétaire, et « oui » à la question « êtes-vous une IA ? ». Pour un achat, l'agent rapporte l'offre, le prix et la référence, et le propriétaire paie.

### Essais prévus

1. Une mission vers le téléphone du propriétaire, qui joue un service proposant deux créneaux puis demandant un numéro de carte. On vérifie : annonce IA, respect de l'ordre, refus de la carte, relecture finale, événement d'agenda, compte rendu, latence.
2. Un vrai service sans enjeu, choisi par le propriétaire, aux heures d'ouverture : demander les horaires du samedi.
3. Une personne que le propriétaire connaît, prévenue avant, sur une vraie démarche.

Chaque étape passe par un ordre de mission, un go, un compte rendu et le verdict du propriétaire avant la suivante.

### Limites et points juridiques

- Échecs relevés par des tests tiers et pris en compte dans la conception : une annonce IA jouée à un serveur vocal au lieu de l'humain qui décroche ensuite, un résumé qui dit « réservé » sans confirmation, un numéro de téléphone inventé, une tentative d'hameçonnage inversé qui demande un numéro de carte et un code reçu par SMS, un commerce qui refuse de traiter avec une IA.
- Administrations françaises (attente longue, vérification d'identité, refus de parler à un tiers) : aucun test publié trouvé. Comme l'agent ne se présente jamais comme le propriétaire, il peut obtenir des informations générales et des rendez-vous, sans accès à un dossier personnel.
- France et Union européenne, d'après des sources lues le 4 novembre 2026, à faire confirmer par un juriste : l'article 50(1) de l'AI Act impose que les personnes sachent qu'elles parlent à une IA, en application depuis le 2 août 2026 ; la CNIL demande d'informer chaque interlocuteur d'un enregistrement en début d'appel ; le démarchage téléphonique des consommateurs sans consentement préalable est interdit depuis le 11 août 2026 ; l'Arcep encadre les numéros que peuvent afficher les systèmes automatisés d'appel, et la question de savoir si un appel ponctuel d'agent en relève n'était pas tranchée dans ce qui a été lu. Ces appels sont des appels de service, et l'agent ne doit jamais servir à prospecter.

## Fait avec Claude Code

Le pipeline a été monté, mesuré et documenté dans des sessions Claude Code menées par l'auteur, et l'agent qui appelle a été conçu de la même manière. La barrière d'envoi reprend la règle qu'il applique à tout message que Claude Code prépare pour lui : rien ne part vers un humain sans le texte intégral affiché et un go explicite qui nomme l'acte.

## Organisation du dépôt

```
pipeline/
  bot.py              bot Pipecat : STT, fin de tour, LLM, TTS, fiche en fin d'appel
  stt_server.py       STT Kyutai en streaming par websocket
  persona.md          persona avec le marqueur {owner}
  bench.py            banc sur fichiers, avec les temps par étape
  test_webrtc.py      client WebRTC qui joue un WAV dans le vrai bot
  make_questions.py   les cinq questions du banc (Kokoro)
  make_scenarios.py   deux scénarios à plusieurs tours (Kokoro)
  scripts/            scripts de démarrage et d'arrêt, variante Windows du lanceur du LLM
  measurements/       relevés du banc du 5 novembre 2026, avec et sans garde
  requirements-*.txt  versions installées dans le montage mesuré
caller/
  mission-order.template.md     ordre de mission, anglais
  mission-order.template.fr.md  ordre de mission, français
```

Les fichiers audio ne sont pas inclus dans le dépôt.

---

État vérifié au 5 novembre 2026.

Ismaël Joffroy Chandoutis, 2026. Code sous licence MIT, textes sous licence CC BY 4.0 ([LICENSE](LICENSE)).

Par [Ismaël Joffroy Chandoutis](https://ismaeljoffroychandoutis.com).
