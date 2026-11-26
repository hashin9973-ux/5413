use engine_core::{
    DataValue, ExperienceEvent, ExperienceJournal, IdempotentAppendOutcome, ObservationFrame,
};
use std::error::Error;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let path: PathBuf = std::env::temp_dir().join(format!(
        "5413-experience-journal-demo-{}.bin",
        std::process::id()
    ));
    if path.exists() {
        fs::remove_file(&path)?;
    }

    let action_event = ExperienceEvent::ActionProposed {
        action_id: "inspect".to_owned(),
        arguments: DataValue::object([(
            "target".to_owned(),
            DataValue::Text("environment-state".to_owned()),
        )]),
    };
    let observation_event = ExperienceEvent::Observation(ObservationFrame {
        environment_id: "example-world".to_owned(),
        session_id: "session-001".to_owned(),
        episode_id: "episode-001".to_owned(),
        sequence: 0,
        observed_at_millis: 1_000,
        schema_id: "world-state".to_owned(),
        schema_version: 1,
        payload: DataValue::object([("status".to_owned(), DataValue::Text("ready".to_owned()))]),
    });

    {
        let mut journal = ExperienceJournal::open(&path)?;
        let first = journal.append_idempotent(
            "episode-001/action/inspect",
            "episode-001",
            action_event.clone(),
        )?;
        let retry =
            journal.append_idempotent("episode-001/action/inspect", "episode-001", action_event)?;
        journal.append_idempotent(
            "episode-001/observation/0",
            "episode-001",
            observation_event,
        )?;

        assert_eq!(first, IdempotentAppendOutcome::Appended { sequence: 0 });
        assert_eq!(
            retry,
            IdempotentAppendOutcome::AlreadyPresent { sequence: 0 }
        );
        println!("synchronized {} unique events", journal.len());
    }

    {
        let mut journal = ExperienceJournal::open(&path)?;
        let replayed_action = journal.append_idempotent(
            "episode-001/action/inspect",
            "episode-001",
            ExperienceEvent::ActionProposed {
                action_id: "inspect".to_owned(),
                arguments: DataValue::object([(
                    "target".to_owned(),
                    DataValue::Text("environment-state".to_owned()),
                )]),
            },
        )?;
        let episode_events = journal.replay_episode("episode-001").count();
        println!(
            "reopened with {} events; episode replay returned {}; retry result: {:?}",
            journal.len(),
            episode_events,
            replayed_action
        );
    }

    fs::remove_file(path)?;
    Ok(())
}
