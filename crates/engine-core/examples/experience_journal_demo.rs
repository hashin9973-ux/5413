use engine_core::{DataValue, ExperienceEvent, ExperienceJournal, ObservationFrame};
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

    {
        let mut journal = ExperienceJournal::open(&path)?;
        journal.append(
            "episode-001",
            ExperienceEvent::ActionProposed {
                action_id: "inspect".to_owned(),
                arguments: DataValue::object([(
                    "target".to_owned(),
                    DataValue::Text("environment-state".to_owned()),
                )]),
            },
        )?;
        journal.append(
            "episode-001",
            ExperienceEvent::Observation(ObservationFrame {
                environment_id: "example-world".to_owned(),
                session_id: "session-001".to_owned(),
                episode_id: "episode-001".to_owned(),
                sequence: 0,
                observed_at_millis: 1_000,
                schema_id: "world-state".to_owned(),
                schema_version: 1,
                payload: DataValue::object([(
                    "status".to_owned(),
                    DataValue::Text("ready".to_owned()),
                )]),
            }),
        )?;
        println!("synchronized {} events", journal.len());
    }

    {
        let journal = ExperienceJournal::open(&path)?;
        let episode_events = journal.replay_episode("episode-001").count();
        println!(
            "reopened journal with {} total events; episode replay returned {} events",
            journal.len(),
            episode_events
        );
    }

    fs::remove_file(path)?;
    Ok(())
}
