from seven.core.di import SevenContainer


def test_di_diagnostic():
    print("\n[1] Creating container")

    container = SevenContainer()

    print("[2] Loading configuration")

    config = container.config_manager()
    config.load()

    print("[3] Resolving SettingsStore")

    store = container.settings_store()

    print("[4] Resolving LoggerService")

    logger = container.logger_service()

    print("[5] Resolving EventBus")

    event_bus = container.event_bus()

    print("[6] Resolving SettingsManager")

    manager = container.settings_manager()

    print("[7] Checking dependencies")

    assert manager._store is store
    assert manager._event_bus is event_bus

    print("[SUCCESS] DI diagnostic completed")

def create_test_container():
    container = SevenContainer()
    container.config_manager().load()
    return container