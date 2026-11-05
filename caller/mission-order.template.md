# Mission order template

Text template, filled for each call and displayed in full before the go. Status on 5 November 2026: no call has been placed with it.

```
MISSION ORDER no. YYYY-MM-DD-xx · name: <short name used in the go>

Goal (one sentence):
Party called: <name of the service or person>
Number to call: <number>   Source of the number: <official site, email of DD/MM...>
Language: French / English
Calling window: <opening hours of the service>
What the agent may say about the owner: <name; file or booking number if useful; nothing else>

The agent MAY ACCEPT:
- slots: <from the owner's calendar, in order of preference>
- maximum budget: <amount incl. tax> (above it: decline or refer to the owner)
- allowed commitments: <e.g. book without a deposit; move an existing appointment>

The agent MUST DECLINE OR REFER TO THE OWNER:
- any payment, any bank detail, any code received by text message or email
- any paid commitment, contract, subscription, signature
- any question about the owner's health, finances or private life outside the mission
- any request to "speak to the account holder": <transfer to the owner / offer a callback>

If nobody answers: <call back later / leave this message: "...">
If asked whether it is an AI: yes, always.
Maximum duration: <x> min (waiting included)
Model: Claude Opus 5.5 / Claude Sonnet 5.5     Voice: <owner's voice / catalogue voice>
Expected report: result, quote of the confirmation, date, price, next steps
```

Opening line, played first and repeated to every new human on the line:

```
Hello, I am <owner>'s AI assistant, with a synthetic voice. I am calling on his behalf about <goal>. This call is recorded.
```

Arming the call: the owner answers `go envoi <name>`, alone in the message. One go, one call, valid for 30 minutes.
