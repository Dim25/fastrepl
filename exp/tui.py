from typing import Union

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Header, Select, RichLog, Button
from textual.containers import Horizontal


class ClassificationApp(App[dict[str, Union[str, int]]]):
    TITLE = "FastREPL"
    SUB_TITLE = "Classification"

    CSS = """
    #log {
        height: 85%;
    }
    """

    def __init__(self, text, classes) -> None:
        super().__init__()
        self.text = text
        self.classes = classes
        self.selected = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", wrap=True)
        with Horizontal():
            yield Select(
                value=self.selected,
                options=((line, i) for i, line in enumerate(self.classes)),
            )
            yield Button("Submit", variant="primary")

    def on_ready(self) -> None:
        log = self.query_one(RichLog)
        lines = self.text.split(". ")
        for line in lines:
            log.write(f"{line}. \n")

    def on_button_pressed(self) -> None:
        self.exit({"text": self.text, "result": self.selected})

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        self.selected = int(str(event.value))


# imdb
TEXT = """
Buffs of the adult western that flourished in the 1950s try and trace its origins to the film that kicked off the syndrome. Of course, we can go back to Howard Hawks\'s Red River (1948) or further still to John Ford\'s My Darling Clementine (1946), but if we want to stick with this single decade, then it has to be one of a couple of films made in that era\'s initial year. One is "The Gunfighter," an exquisitely grim tale of a famed gunslinger (Ringo) facing his last shootout. Another from that same year is "Winchester \'73," and it\'s worth noting that Millard Mitchell appears in both as grim, mustached, highly realistic range riders. In The Gunfighter, he\'s the town marshal expected to arrest Ringo but once rode with him in an outlaw gang. In Winchester, he\'s the sidekick to Jimmy Stewart, a kind of Horatio to Stewart\'s Hamlet in this epic/tragic tale. The plot is simple enough: Stewart\'s lonesome cowpoke wins a remarkable Winchester in a shooting match, beating the meanest man in the west (Stephen McNally), who is actually his own brother and caused the death of their father. When the brother steals the gun, Stewart and Mitchell go after him in a cowboy odyssey that takes them all across the frontier, meeting up with both outlaws and Indians. (In one wonderful bit, two future stars - Rock Hudson and Tony Curtis - play an Indian chief and a U.S. cavalry soldier - during a well staged pitched-battle. Dan Duryea steals the whole show as a giggling outlaw leader, while Shelly Winters, just before she began to gain weight, is fine as the shady lady who ties all the plots together. Today, filmmakers would go on for about four hours to bring such an ambitious idea to the screen, but Anthony Mann does so in an extremely economical amount of time, with not a minute wasted. Such western legends as Bat Masterson and Wyatt Earp (terrifically played by Will Geer) make brief appearances, adding to the historicity as well as the epic nature. The final battle between good and bad brothers, high atop a series of jutting rock canyons, is now legendary among western buffs. It\'s also worth noting that Stewart, however much associated he became with western films, does what is actually his first western leading man role here - yes, he was in Destry Rides Again eleven years earlier, but was cast in that comedy spoof because he seemed so WRONG for westerns!
"""


if __name__ == "__main__":
    app = ClassificationApp(text=TEXT, classes=["POSITIVE", "NEGATIVE"])
    result = app.run()
    print(result)
