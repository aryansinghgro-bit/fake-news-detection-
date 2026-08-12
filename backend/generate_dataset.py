"""Generate a realistic labeled fake-news dataset for TruthScan AI.

Creates backend/data/news.csv with columns (title, text, label).

The dataset is designed so that REAL and FAKE articles share stylistic
overlap — both use professional headlines and body text — forcing the
TF-IDF + Logistic Regression pipeline to learn genuine lexical patterns
rather than trivially memorizing surface markers like ALL-CAPS or
exclamation marks.

REAL articles: attributed sources, hedged language, verifiable-sounding
facts, neutral tone, conventional journalistic structure.

FAKE articles: fabricated attributions, definitive claims without
evidence, emotional manipulation, unsubstantiated assertions, vague
"sources say" framing, conspiracy-adjacent vocabulary.

Run once before train_model.py:
    python generate_dataset.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

DATA_PATH = Path(__file__).resolve().parent / "data" / "news.csv"

# ----------------------------------------------------------------------- #
# REAL article templates — measured, attributed, journalistic
# ----------------------------------------------------------------------- #
REAL_TEMPLATES = [
    (
        "City Council Approves Budget for New Public Library",
        "The city council voted 6-2 on Tuesday to approve a 4.2 million dollar budget for the construction of a new public library in the downtown district. Mayor Jane Doe said the facility will serve approximately 30,000 residents and include a community meeting room and expanded digital archives. Construction is expected to begin in March and conclude by late next year. Council members who opposed the measure cited concerns about long-term operating costs.",
    ),
    (
        "Federal Reserve Holds Interest Rates Steady",
        "The Federal Reserve announced on Wednesday that it would maintain its benchmark interest rate at the current level, citing stable inflation and moderate economic growth. Officials said the decision reflects continued progress toward the central bank's employment and price-stability goals. Economists surveyed had broadly expected the hold. The statement noted that future adjustments would depend on incoming economic data.",
    ),
    (
        "Researchers Publish Study on Sleep and Memory",
        "A study published in the Journal of Neuroscience examined the relationship between sleep duration and memory consolidation in adults. Researchers at State University followed 240 participants over 12 weeks and reported that those sleeping fewer than six hours per night showed measurably lower performance on recall tasks. The authors cautioned that the findings are correlational and that further research is needed to establish causation.",
    ),
    (
        "Local School District Reports Graduation Rate Increase",
        "The Riverside School District announced that its four-year graduation rate rose to 89 percent this year, up from 84 percent the previous year. Superintendent Mark Lee attributed the improvement to expanded tutoring programs and smaller class sizes in ninth-grade mathematics. The district plans to present detailed figures at the school board meeting next month.",
    ),
    (
        "NASA Announces Date for Next Satellite Launch",
        "NASA confirmed that its next environmental monitoring satellite is scheduled to launch from Vandenberg Space Force Base on October 14. The spacecraft will collect data on ocean surface temperatures and atmospheric moisture. Mission managers said the launch window opens at 9:02 a.m. local time and that weather conditions currently appear favorable.",
    ),
    (
        "Senate Committee Advances Infrastructure Bill",
        "The Senate Commerce Committee advanced a bipartisan infrastructure bill on Thursday with a 14-12 vote. The legislation proposes 120 billion dollars in funding for road and bridge repairs over five years. Senator Alan Brooks, a co-sponsor, said the bill will now move to the full Senate for debate. Several amendments regarding funding sources remain under discussion.",
    ),
    (
        "Tech Company Reports Quarterly Earnings",
        "Nimbus Technologies reported quarterly revenue of 3.1 billion dollars, a 7 percent increase from the same period last year. Chief Executive Officer Sarah Chen said growth was driven by demand for the company's cloud services. The company also announced a planned expansion of its data center operations in two states. Shares closed up 2.4 percent.",
    ),
    (
        "Hospital Opens New Pediatric Wing",
        "Mercy General Hospital opened a new pediatric wing on Monday, adding 40 beds and a dedicated intensive care unit for children. Hospital officials said the 18 million dollar project was funded through a combination of state grants and private donations. Dr. Emily Ross, head of pediatrics, said the facility will reduce wait times for families in the region.",
    ),
    (
        "Agricultural Report Shows Wheat Yield Increase",
        "The Department of Agriculture released its annual crop report, showing a 5 percent increase in national wheat yields compared with last year. Analysts attributed the gain to favorable rainfall in the Plains states. The report also noted a slight decline in corn output due to early-season drought conditions in parts of the region.",
    ),
    (
        "Transportation Authority Extends Subway Service Hours",
        "The Metropolitan Transportation Authority voted to extend weekend subway service by one hour beginning next month. Authority chairperson Robert Kim said the change responds to rider feedback and is expected to benefit approximately 200,000 weekend passengers. The agency will monitor ridership data over the following six months.",
    ),
    (
        "University Receives Grant for Renewable Energy Research",
        "State University announced a 6 million dollar grant from the Department of Energy to support research into next-generation solar cells. The project will be led by the Department of Materials Science and will involve graduate students and postdoctoral researchers. The university said the funding covers a three-year period.",
    ),
    (
        "Weather Service Issues Heat Advisory for Southern Counties",
        "The National Weather Service issued a heat advisory for three southern counties, forecasting temperatures above 100 degrees through Friday. Residents were advised to limit outdoor activity and check on elderly neighbors. Cooling centers will be open at community facilities in each affected county.",
    ),
    (
        "Court Rules in Favor of City in Zoning Dispute",
        "A state appeals court ruled in favor of the city in a long-running zoning dispute involving a proposed residential development. The three-judge panel found that the city's planning commission had followed proper procedure. An attorney for the developer said they are reviewing the decision and have not decided whether to appeal.",
    ),
    (
        "Public Health Officials Report Decline in Flu Cases",
        "Public health officials reported a 15 percent decline in confirmed influenza cases this season compared with last season. Dr. Patricia Ng of the county health department credited increased vaccination rates and earlier public awareness campaigns. She noted that flu activity typically peaks in February and urged residents to continue preventive measures.",
    ),
    (
        "New Bridge Construction Reaches Milestone",
        "Construction of the replacement Harbor Bridge reached a milestone this week as crews installed the main suspension cables. The project, which began two years ago, is now 60 percent complete according to the state transportation department. The bridge is expected to open to traffic next summer.",
    ),
    (
        "International Trade Agreement Signed at Summit",
        "Representatives from 12 nations signed a regional trade agreement at the annual economic summit on Friday. The agreement reduces tariffs on selected agricultural and manufactured goods and establishes a framework for ongoing negotiations. Officials said implementation will occur in phases over the next three years.",
    ),
    (
        "Veterans Affairs Hospital Adds Mental Health Staff",
        "The regional Veterans Affairs hospital announced the addition of 12 mental health counselors to its staff. Hospital director James Patterson said the expansion will reduce appointment wait times for veterans seeking counseling. The positions were funded through a federal appropriation approved earlier this year.",
    ),
    (
        "Environmental Group Releases Water Quality Report",
        "The Watershed Alliance released its annual water quality report, finding that 78 percent of tested river sites met federal cleanliness standards, up from 71 percent last year. The report attributed the improvement to upgraded wastewater treatment facilities. The group recommended continued monitoring of industrial discharge points.",
    ),
    (
        "City Approves Grant for Small Business Relief",
        "The city approved a 2 million dollar grant program to provide relief to small businesses affected by recent construction on Main Street. Eligible businesses can apply for grants of up to 10,000 dollars. Applications will be reviewed by a committee of local business owners and city officials.",
    ),
    (
        "School Board Adopts New Science Curriculum",
        "The school board voted 5-3 to adopt a new science curriculum for grades six through eight. The curriculum emphasizes hands-on laboratory work and was developed over 18 months by a committee of teachers and university faculty. Implementation is planned for the next academic year.",
    ),
    (
        "Pharmaceutical Company Recalls Batch of Blood Pressure Medication",
        "Vance Pharmaceuticals announced a voluntary recall of one production lot of its blood pressure medication after routine testing identified a labeling discrepancy. The company said no adverse health effects have been reported and that the recall is precautionary. Patients are advised to consult their pharmacist for guidance.",
    ),
    (
        "Governor Signs Police Reform Bill Into Law",
        "Governor Maria Torres signed a police reform bill into law on Thursday, establishing new training requirements and a statewide database of officer disciplinary records. The legislation passed both chambers with bipartisan support. Civil rights groups called the bill a step forward while some law enforcement organizations expressed concerns about implementation costs.",
    ),
    (
        "Study Links Moderate Exercise to Improved Cardiovascular Health",
        "Researchers at Johns Hopkins University published findings suggesting that 150 minutes of moderate exercise per week is associated with a 20 percent reduction in cardiovascular events among adults over 50. The study tracked 8,000 participants for six years. Authors noted that the observational design limits causal conclusions.",
    ),
    (
        "Electric Vehicle Sales Reach Record High in Third Quarter",
        "Electric vehicle sales reached a record 320,000 units in the third quarter, representing 9 percent of all new vehicle sales, according to industry data. Analysts attributed the increase to expanded model availability and federal tax incentives. Several manufacturers announced plans to increase production capacity next year.",
    ),
]

# ----------------------------------------------------------------------- #
# FAKE article templates — fabricated, sensational, manipulative
# ----------------------------------------------------------------------- #
FAKE_TEMPLATES = [
    (
        "Secret Government Program Controls Weather, Insiders Reveal",
        "A whistleblower has come forward with claims that a secret government agency has been controlling the weather for decades using hidden technology. According to anonymous sources, powerful elites are manipulating hurricanes and droughts to profit from global markets. The mainstream media refuses to cover this story, but the truth is finally coming out. Sources say the technology has been operational since the 1990s and that the public has been completely unaware.",
    ),
    (
        "Miracle Cure Banned by Pharmaceutical Companies, Sources Claim",
        "A simple herb can supposedly cure every disease known to humanity, but pharmaceutical companies are desperately trying to suppress it, according to claims circulating online. A secret formula, hidden for centuries, has allegedly been used by a remote village where no one ever gets sick. Insiders claim the herb has been banned because it would destroy the pharmaceutical industry. These claims have not been verified by any medical authority.",
    ),
    (
        "Leaked Documents Prove Government Has Been Hiding Alien Contact",
        "Leaked documents have supposedly confirmed that extraterrestrial beings have been in contact with government officials for over 70 years. Anonymous researchers claim a secret military program has been reverse-engineering alien technology since the 1950s. The mainstream media is completely silent about this revelation. Sources say the government has signed agreements with multiple alien species and the truth is bigger than anyone could imagine.",
    ),
    (
        "Hollywood Insiders Reveal Secret Society Controls Entertainment Industry",
        "Shocking new rumors suggest that Hollywood celebrities are involved in a secret society that controls the entertainment industry. Anonymous insiders claim that only those who join the society are allowed to become famous. Sources say anyone who tries to reveal the truth is silenced. The allegations are unverified and the mainstream media is reportedly suppressing the story.",
    ),
    (
        "Hidden Camera Captures Doctors Admitting Vaccine Contains Microchip",
        "A hidden camera investigation has allegedly captured doctors admitting that vaccines contain secret microchips designed to track every citizen. The footage, which has not been verified, supposedly shows medical professionals discussing a global surveillance plot. The mainstream media is refusing to report on the video. Sources claim the microchips are part of a plan to monitor and control the population.",
    ),
    (
        "Anonymous Hacker Proves Millions of Votes Were Changed in Election",
        "An anonymous hacker has supposedly uncovered proof that millions of votes in the recent election were secretly changed by a shadowy organization. The claims, which have not been verified by any official source, suggest that the entire election was rigged from the start. The mainstream media is ignoring the story. Sources who cannot be named say the hacking was orchestrated by powerful elites.",
    ),
    (
        "Former NASA Employee Claims All Space Missions Have Been Faked",
        "A former NASA employee has allegedly come forward to claim that all space missions have been faked and the Earth is actually flat. According to anonymous sources, every photograph from space has been created in a studio. The mainstream media continues to promote what the source calls a lie. Insiders say astronauts are actors and rockets never actually leave the atmosphere.",
    ),
    (
        "Former Banker Exposes Secret Society Controlling All World Banks",
        "A former banker has supposedly revealed that a secret society controls every central bank in the world and manipulates the global economy. Anonymous sources claim that a small group of elites meets in secret to decide the fate of nations. The mainstream media refuses to investigate. Insiders say ordinary people are being kept in poverty deliberately while the elites hoard wealth.",
    ),
    (
        "Government Official Allegedly Admits Chemtrails Are Chemical Spraying",
        "A government official has allegedly admitted that the white trails behind airplanes are actually chemicals being sprayed to control the population. The claims, which have no verifiable evidence, suggest a massive covert operation has been ongoing for decades. Mainstream scientists are described as part of the cover-up. Sources say the chemicals are designed to alter human behavior.",
    ),
    (
        "Independent Researchers Claim 5G Towers Spread Deadly Virus",
        "Independent researchers claim that 5G cell towers are spreading a deadly virus and that scientists who speak out are being silenced. According to anonymous sources, the telecommunications industry knows the truth but is hiding it to protect profits. The mainstream media will not report on the findings. Insiders say the virus was engineered to coincide with the rollout of 5G technology.",
    ),
    (
        "Music Insiders Claim Hidden Messages in Songs Prove Cult Worship",
        "Music industry insiders have allegedly revealed that hidden messages in popular songs prove that artists are part of a secret cult. Anonymous sources claim that only those who pledge loyalty to the cult are allowed to reach the top of the charts. The mainstream media is described as complicit in the cover-up. Sources say the messages are embedded using secret technology.",
    ),
    (
        "Explorers Claim Mysterious Pyramid Found Hidden in Antarctica",
        "A team of explorers has supposedly discovered a mysterious pyramid in Antarctica, but governments around the world are working together to hide the discovery. According to anonymous sources, the pyramid contains evidence of an ancient civilization that predates all known history. Mainstream scientists are refusing to investigate. Insiders say the discovery would rewrite history and that is why it is being suppressed.",
    ),
    (
        "Former Aide Claims World Billionaires Are Reptilian Shapeshifters",
        "A former aide to a prominent billionaire has allegedly come forward to claim that the world's richest people are actually reptilian shapeshifters. The claims, which cannot be verified, suggest that a race of reptile beings has infiltrated the highest levels of government and business. The mainstream media reportedly dismisses the story as a joke. Sources say the reptilians have been in control for centuries.",
    ),
    (
        "UFO Researchers Claim Secret Moon Base Visible in NASA Photographs",
        "UFO researchers claim to have discovered a secret base on the moon in photographs released by NASA. According to anonymous experts, the base was built by a shadowy organization and has been operational for decades. NASA has refused to comment on the allegations. Sources say the photos were accidentally released and quickly removed from public archives.",
    ),
    (
        "Whistleblower Claims Energy Companies Murdered Free Energy Inventor",
        "An anonymous whistleblower claims that a scientist invented a machine that produces free energy, but was murdered by energy companies to protect their profits. The claims have no verifiable evidence, but sources say the inventor's notes have been hidden in a secret location. The mainstream media refuses to investigate the story. Insiders say the technology would have eliminated the need for oil and gas.",
    ),
    (
        "Self-Proclaimed Expert Claims Sinkholes Are Secret Government Tunnels",
        "A self-proclaimed expert claims that the mysterious sinkholes appearing across the country are actually secret tunnels being built by the government. According to anonymous sources, the tunnels connect underground military bases. Mainstream geologists say the sinkholes are natural, but insiders insist the truth is being hidden. Sources say the tunnels are part of a plan to relocate elites.",
    ),
    (
        "Person Claiming to Be Time Traveler Warns of Imminent Catastrophe",
        "A person claiming to be a time traveler from the year 2087 has supposedly warned of an imminent catastrophe that will destroy civilization. The claims, which are impossible to verify, suggest that a secret event will occur within months. The mainstream media is ignoring the warning. Anonymous sources say the time traveler has provided detailed predictions that have allegedly come true.",
    ),
    (
        "Researchers Claim Hidden Bible Code Predicts Exact Date of World End",
        "Researchers claim to have discovered a hidden code in the Bible that predicts the exact date of the end of the world. According to anonymous sources, the code was deciphered using advanced computer algorithms. Mainstream religious scholars dismiss the claims, but insiders say the evidence is undeniable. Sources say the date is approaching and the public has not been warned.",
    ),
    (
        "Health Guru Claims One Weird Food Melts Belly Fat Overnight",
        "A health guru claims that one weird food can melt belly fat overnight, and that doctors are furious because it destroys their business. The claims have no scientific backing, but anonymous sources say the food has been kept secret by the medical industry. Mainstream nutritionists dismiss the story. Insiders say the trick works for everyone and requires no exercise.",
    ),
    (
        "Astronomers Supposedly Received Alien Warning Signal from Deep Space",
        "Astronomers have supposedly received a mysterious signal from deep space that they claim is a warning from an alien civilization. According to anonymous sources, the signal contains a message about an impending event. Mainstream scientists are refusing to acknowledge the finding. Insiders say the government has known about the signal for years and is hiding it from the public.",
    ),
    (
        "Documents Allegedly Prove Government Created Virus in Secret Lab",
        "Leaked documents have supposedly confirmed that a recent virus was deliberately created in a secret government laboratory. According to anonymous sources, the virus was engineered as part of a population control plan. The mainstream media is described as complicit in the cover-up. Sources say the evidence has been suppressed by powerful interests and the public deserves the truth.",
    ),
    (
        "Insider Claims Stock Market Is Secretly Controlled by Single Family",
        "A former financial regulator has allegedly revealed that the entire stock market is secretly controlled by a single powerful family. According to anonymous sources, every major market movement is orchestrated from behind the scenes. The mainstream media refuses to investigate. Insiders say ordinary investors have no chance of profiting because the system is rigged.",
    ),
    (
        "Former Scientist Claims Fluoride in Water Is Mind Control Chemical",
        "A former government scientist has allegedly admitted that fluoride in public water supplies is actually a mind control chemical. The claims, which contradict decades of public health research, suggest a covert program to make the population docile. Mainstream scientists are described as part of the conspiracy. Sources say the truth has been hidden since the 1950s.",
    ),
    (
        "Anonymous Source Claims Cancer Cure Found But Suppressed by Industry",
        "An anonymous source has claimed that a cure for cancer was discovered years ago but has been deliberately suppressed by the medical industry to protect profits. According to the claims, a secret treatment is being used only by the elite while ordinary patients are denied access. The mainstream media reportedly refuses to cover the story. These claims have no verifiable evidence.",
    ),
]


# ----------------------------------------------------------------------- #
# Filler sentences — shared between both classes to add lexical noise
# ----------------------------------------------------------------------- #
FILLER = [
    "The report was published on Tuesday.",
    "Officials declined to comment further.",
    "The findings have not been independently verified.",
    "The announcement was made at a press conference.",
    "The full report is available on the department website.",
    "The results will be presented at a conference next month.",
    "The investigation is ongoing according to sources.",
    "The decision was reached after lengthy discussions.",
    "The project is expected to take several months.",
    "The proposal has received mixed reactions from the public.",
    "The committee will meet again next week to discuss the matter.",
    "The data was collected over a period of two years.",
    "The statement was released on Friday afternoon.",
    "The program is funded through a combination of public and private sources.",
    "The outcome remains uncertain at this time.",
    "The policy takes effect at the beginning of next year.",
    "The organization has not yet issued a formal response.",
    "The plan is subject to approval by the governing board.",
    "The report recommends further study of the issue.",
    "The meeting lasted approximately three hours.",
    "Additional details are expected to be released in the coming weeks.",
    "The matter has drawn attention from several advocacy groups.",
    "Representatives from both parties participated in the discussions.",
    "The findings are preliminary and subject to peer review.",
    "Local residents have expressed a range of opinions on the proposal.",
]


# Synonyms for paraphrasing — each entry maps a word to a list of replacements.
SYNONYMS: dict[str, list[str]] = {
    "announced": ["said", "stated", "declared", "confirmed", "reported"],
    "said": ["stated", "noted", "explained", "indicated", "remarked"],
    "reported": ["stated", "indicated", "showed", "revealed", "documented"],
    "confirmed": ["stated", "announced", "verified", "established", "affirmed"],
    "claimed": ["alleged", "asserted", "stated", "suggested", "declared"],
    "allegedly": ["supposedly", "reportedly", "purportedly", "apparently"],
    "supposedly": ["allegedly", "reportedly", "purportedly", "apparently"],
    "anonymous": ["unnamed", "unidentified", "secret", "undisclosed"],
    "secret": ["hidden", "clandestine", "covert", "classified"],
    "sources": ["insiders", "officials", "informants", "contacts", "individuals"],
    "officials": ["authorities", "representatives", "spokespeople", "administrators"],
    "researchers": ["scientists", "investigators", "experts", "analysts", "scholars"],
    "study": ["report", "analysis", "investigation", "examination", "research"],
    "report": ["study", "analysis", "document", "assessment", "findings"],
    "findings": ["results", "conclusions", "outcomes", "discoveries", "observations"],
    "government": ["administration", "authorities", "officials", "regime"],
    "company": ["firm", "corporation", "organization", "business", "enterprise"],
    "industry": ["sector", "business", "trade", "market", "field"],
    "program": ["initiative", "project", "scheme", "effort", "operation"],
    "project": ["initiative", "effort", "undertaking", "venture", "program"],
    "public": ["citizens", "people", "population", "residents", "community"],
    "media": ["press", "news outlets", "journalism", "broadcasters", "coverage"],
    "technology": ["tech", "machinery", "equipment", "systems", "innovation"],
    "health": ["medical", "wellness", "healthcare", "medicine", "treatment"],
    "medical": ["health", "clinical", "healthcare", "therapeutic", "medicinal"],
    "doctors": ["physicians", "medical professionals", "medics", "practitioners"],
    "scientists": ["researchers", "experts", "academics", "specialists", "investigators"],
    "evidence": ["proof", "data", "documentation", "testimony", "indication"],
    "claims": ["assertions", "allegations", "statements", "charges", "declarations"],
    "truth": ["reality", "facts", "actual events", "what really happened"],
    "hidden": ["secret", "concealed", "buried", "suppressed", "covered up"],
    "suppressed": ["hidden", "concealed", "buried", "covered up", "censored"],
    "mainstream": ["corporate", "establishment", "major", "traditional", "conventional"],
    "powerful": ["influential", "wealthy", "elite", "prominent", "high-ranking"],
    "elites": ["insiders", "powerful figures", "those in control", "the wealthy"],
    "decades": ["years", "a long time", "many years", "generations"],
    "centuries": ["hundreds of years", "generations", "a very long time"],
    "huge": ["massive", "enormous", "vast", "major", "significant"],
    "massive": ["huge", "enormous", "vast", "major", "significant"],
    "dangerous": ["hazardous", "risky", "deadly", "harmful", "threatening"],
    "deadly": ["dangerous", "lethal", "fatal", "hazardous", "deadly"],
    "mysterious": ["unexplained", "strange", "puzzling", "mysterious", "unknown"],
    "shocking": ["stunning", "surprising", "startling", "remarkable", "alarming"],
    "banned": ["prohibited", "outlawed", "forbidden", "restricted", "blocked"],
    "cure": ["treatment", "remedy", "solution", "therapy", "medicine"],
    "disease": ["illness", "sickness", "condition", "ailment", "disorder"],
    "virus": ["pathogen", "infection", "disease", "bug", "illness"],
    "alien": ["extraterrestrial", "otherworldly", "foreign", "off-world"],
    "secret": ["hidden", "clandestine", "covert", "classified", "undisclosed"],
    "warned": ["cautioned", "alerted", "advised", "forewarned", "notified"],
    "control": ["manipulate", "direct", "command", "oversee", "govern"],
    "profit": ["financial gain", "money", "revenue", "earnings", "income"],
    "rigged": ["fixed", "manipulated", "orchestrated", "staged", "set up"],
    "exposed": ["revealed", "uncovered", "disclosed", "brought to light", "leaked"],
    "revealed": ["exposed", "uncovered", "disclosed", "showed", "demonstrated"],
    "invented": ["created", "developed", "designed", "built", "devised"],
    "murdered": ["killed", "assassinated", "eliminated", "silenced", "put to death"],
    "silenced": ["muzzled", "suppressed", "shut down", "intimidated", "threatened"],
    "decades": ["years", "a long time", "many years", "a generation"],
    "completely": ["entirely", "totally", "fully", "wholly", "absolutely"],
    "supposedly": ["allegedly", "reportedly", "purportedly", "apparently", "claiming to be"],
}


def _paraphrase(text: str) -> str:
    """Apply random synonym substitution and sentence reordering to produce
    a genuinely different but semantically equivalent variant of the text."""
    words = text.split()
    result: list[str] = []
    for word in words:
        clean = word.lower().rstrip(".,;:!?")
        if clean in SYNONYMS and random.random() < 0.5:
            replacement = random.choice(SYNONYMS[clean])
            # Preserve capitalization of first word
            if word[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result.append(replacement)
        else:
            result.append(word)
    text = " ".join(result)

    # Reorder sentences 50% of the time (keeps first sentence in place for coherence)
    sentences = text.split(". ")
    if len(sentences) > 2 and random.random() < 0.5:
        first = sentences[0]
        rest = sentences[1:]
        random.shuffle(rest)
        text = first + ". " + ". ".join(rest)

    return text


def expand_templates(templates: list[tuple[str, str]], target: int) -> list[tuple[str, str]]:
    """Expand templates into a larger dataset by paraphrasing each instance
    with synonym substitution and sentence reordering, then appending
    neutral filler sentences. This produces genuinely different texts that
    share the same core signal, forcing the model to learn lexical patterns
    rather than memorizing identical strings."""
    out: list[tuple[str, str]] = []
    while len(out) < target:
        title, body = random.choice(templates)
        # Paraphrase both title and body for genuine variation
        ptitle = _paraphrase(title) if random.random() < 0.4 else title
        pbody = _paraphrase(body)
        extra = " ".join(random.sample(FILLER, k=random.randint(1, 3)))
        out.append((ptitle, f"{pbody} {extra}"))
    return out[:target]


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    real = expand_templates(REAL_TEMPLATES, 140)
    fake = expand_templates(FAKE_TEMPLATES, 140)
    rows = [(t, b, "REAL") for t, b in real] + [(t, b, "FAKE") for t, b in fake]
    random.shuffle(rows)

    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "label"])
        writer.writerows(rows)

    print(f"[INFO] Generated {len(rows)} samples -> {DATA_PATH}")
    print(f"       REAL: {len(real)} | FAKE: {len(fake)}")


if __name__ == "__main__":
    main()
