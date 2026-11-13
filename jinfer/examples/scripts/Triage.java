///usr/bin/env jbang "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.4.0-rc1@pom
//DEPS com.qxotic:jinfer-decision com.qxotic:jinfer-lfm3 com.qxotic:jinfer-hub
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Route a support ticket with probabilities: one forward pass, nothing generated.
//   jbang Triage.java "I was charged twice and I want my money back today."
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.decision.Answer;
import com.qxotic.jinfer.decision.Answers;
import com.qxotic.jinfer.decision.Decisions;
import com.qxotic.jinfer.decision.Question;
import com.qxotic.jinfer.hub.ModelStore;

import java.lang.foreign.Arena;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;

public class Triage {

    private static final String DEFAULT_MODEL = "LiquidAI/d1-2B-GGUF:Q8_0";

    /**
     * The model answers with a probability per option; these are the numbers an application acts
     * on. A threshold tuned for this checkpoint does not carry to another one.
     */
    public static void main(String[] args) throws Exception {
        String ticket = args.length > 0
                ? args[0]
                : "I was charged twice for my order last week and I want my money back today.";
        String modelRef = args.length > 1 ? args[1] : DEFAULT_MODEL;

        SequencedMap<String, String> teams = new LinkedHashMap<>();
        teams.put("billing", "Charges, invoices and refunds");
        teams.put("shipping", "Delivery and parcels");
        teams.put("technical", "The product does not work");
        teams.put("account", "Login, password and profile");

        Question.Choice<String> team = Question.choice("Which team should handle this?", teams);
        Question.Predicate refund = Question.predicate(
                "Is the customer asking for a refund?", "asks for money back", "asks anything else");
        Question.Rating<String> urgency = Question.rating(
                "How urgent is this?", "Can wait", "This week", "Today", "Right now");

        try (Arena arena = Arenas.newCrossThread()) {
            ModelStore hub = ModelStore.standard();
            DecisionModel model = Models.loadDecisionModel(hub.resolve(modelRef), arena);
            // ask takes the questions the way d1 was trained to: the ticket read once for all
            // of them, an answer free to shift with its neighbours. askIndependently answers
            // each as if asked alone. Both are right: pick the one the application needs.
            Decisions decisions = Decisions.of(model);
            try (ContextState state = model.newState(2048)) {
                print(decisions.ask(state, ticket, team, refund, urgency), team, refund, urgency);
            }
        }
    }

    private static void print(
            Answers answers,
            Question.Choice<String> team,
            Question.Predicate refund,
            Question.Rating<String> urgency) {
        Answer.Choice<String> routed = answers.get(team);
        System.out.printf("team     %-10s %.4f%n", routed.selectedOption(), routed.confidence());
        for (Map.Entry<String, Float> option : probabilities(routed).entrySet()) {
            System.out.printf("         %-10s %.4f%n", option.getKey(), option.getValue());
        }

        Answer.Predicate moneyBack = answers.get(refund);
        System.out.printf("refund   %-10s %.4f%n", moneyBack.isYes() ? "yes" : "no", moneyBack.yes());

        Answer.Rating<String> when = answers.get(urgency);
        System.out.printf(
                "urgency  %-10s %.4f  (mean %.2f, median %s)%n",
                when.selectedLevel(),
                when.confidence(),
                when.mean(),
                when.medianLevel());

        // the one number an application usually gates on: how sure the model is
        System.out.printf("%nsure enough to auto-route: %s%n", routed.confidence() >= 0.9f);
    }

    private static Map<String, Float> probabilities(Answer.Choice<String> answer) {
        Map<String, Float> byOption = new LinkedHashMap<>();
        List<String> options = answer.options();
        for (int i = 0; i < options.size(); i++) {
            byOption.put(options.get(i), answer.probabilities().get(i));
        }
        return byOption;
    }
}
