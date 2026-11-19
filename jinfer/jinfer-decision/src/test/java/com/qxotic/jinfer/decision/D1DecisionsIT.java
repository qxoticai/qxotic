package com.qxotic.jinfer.decision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The typed questions answered by a real decision model, Liquid d1-2B. */
@Tag("integration")
final class D1DecisionsIT {

    enum Team {
        BILLING,
        SHIPPING,
        TECHNICAL
    }

    @Test
    void aTicketIsRoutedAndRated() throws Exception {
        String ticket =
                "Customer: I was charged twice for my order last week and I want my money back"
                        + " today. This is the third time I am writing.";
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 =
                    Models.loadDecisionModel(
                            TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf"), arena);
            Decisions decisions = Decisions.of(d1);
            try (ContextState state = d1.newState(1024)) {
                var refund = Question.predicate("Is the customer asking for a refund?");
                var team = Question.choice("Which team should handle this?", Team.class);
                var urgency =
                        Question.rating(
                                "How urgent is this?",
                                "Can wait",
                                "This week",
                                "Today",
                                "Right now");

                assertEquals(
                        Team.BILLING,
                        decisions
                                .ask(state, ticket, team)
                                .selectedOption()); // one question, its answer

                // together: one reading of the ticket for the three questions
                Answers together = decisions.ask(state, ticket, refund, team, urgency);
                assertTrue(together.get(refund).isYes());
                assertEquals(Team.BILLING, together.get(team).selectedOption());

                Answers answers = decisions.askIndependently(state, ticket, refund, team, urgency);
                assertTrue(answers.get(refund).isYes(), "refund " + answers.get(refund));
                assertEquals(Team.BILLING, answers.get(team).selectedOption());
                assertTrue(answers.get(team).confidence() > 0.8f, "team " + answers.get(team));
                assertTrue(
                        answers.get(urgency).selectedIndex() >= 2,
                        "urgency " + answers.get(urgency));
            }
        }
    }

    enum Shown {
        CAT,
        DOG,
        DICE
    }

    /** A picture and its text, through the model's vision tower. */
    @Test
    void aPictureIsJudged() throws Exception {
        Path images = TestModels.fixture("lfm3/images");
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 =
                    Models.loadDecisionModel(
                            TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf"),
                            arena,
                            Map.of(
                                    "media",
                                    TestModels.require(
                                            "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf")));
            Decisions decisions = Decisions.of(d1);
            try (ContextState state = d1.newState(0)) {
                var shown = Question.choice("What does the picture show?", Shown.class);
                var cat = Question.predicate("Is there a cat in the picture?");
                List<Content> photo =
                        List.of(
                                new Content.Media(ImageCodec.load(images.resolve("dice.png"))),
                                new Content.Text("A photo sent by a customer."));
                assertEquals(Shown.DICE, decisions.ask(state, photo, shown).selectedOption());
                Answers answers = decisions.ask(state, photo, shown, cat);
                assertEquals(Shown.DICE, answers.get(shown).selectedOption());
                assertTrue(!answers.get(cat).isYes(), "cat " + answers.get(cat));
            }
        }
    }
}
