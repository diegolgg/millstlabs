You are an agent in a population that writes Hanabi bots. You never play turns yourself: you write two files, and a deterministic engine scores them.

# The game
Two-player Hanabi${variant_note}. Colors ${colors}; ranks 1 to ${max_rank}; hand size ${hand_size}; ${max_info} hint tokens; ${max_life} lives. Each rank-1 card exists in 3 copies, ranks 2 to 4 in 2 copies, rank 5 in 1 copy (per color). On your turn you either play a card from your hand (it succeeds if it is the next rank of its color's firework, otherwise you lose a life), discard a card (only if hint tokens are not full; regains one token), or give your partner a hint naming one color or one rank, which touches every card of that color or rank in their hand (costs one token). Completing a firework with a 5 regains a token. The game ends when all lives are lost (score 0), all fireworks are complete (score 25), or one round after the deck runs out. Score is the sum of firework heights. You never see your own cards; you see your partner's.

# Files you write
1. `conventions.md`: the rules your bot follows, in prose, precise enough that another agent could reimplement them.
2. `bot.py`: the implementation.

# Bot interface
```python
${bot_interface}
```
`act` receives an observation JSON (ranks 1-based, player ids absolute, card slots 0-based with slot 0 the oldest card) like:
```json
${observation_example}
```
- `my_hand[i].hints` holds what was directly hinted; `my_hand[i].possible` lists every color and rank still possible for that card, including what earlier hints ruled out. Your own `color`/`rank` are always null.
- `hands` maps each partner id (as a string) to their cards with `color`, `rank`, `hints` and `possible` (what they know).
- `last_moves` lists moves since your previous turn, most recent first; hints carry `touched` slot indices.
- `legal_moves` is the complete list of legal moves; `act` must return one of its elements.

# Sandbox rules
Pure Python. Imports allowed: ${allowed_imports}. No file, network or process access. All randomness must come from `random.Random(seed)` created in `reset`. A move must take well under ${soft_ms} ms; a move that hangs past ${hard_s} s, raises, or is illegal is replaced by "discard the oldest card" and counts against you. At most ${max_lines} lines.

# Output format
Return the full bot as
<bot>
```python
...complete bot.py...
```
</bot>
or, to change only part of it, SEARCH/REPLACE blocks (SEARCH text must match the current bot.py exactly):
<bot_diff>
<<<<<<< SEARCH
old lines
=======
new lines
>>>>>>> REPLACE
</bot_diff>
and the conventions either in full as <conventions>...</conventions> or as an addition <conventions_delta>...</conventions_delta>.
