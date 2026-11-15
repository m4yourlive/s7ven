"""Tests for database models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from seven.core.database.base import Base
from seven.core.database.models import Conversation, Memory, Message


@pytest.fixture
def db_session():
    """Provide an isolated in-memory database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )

    with session_factory() as session:
        yield session

    engine.dispose()


class TestConversationModel:
    def test_default_values(self, db_session):
        conversation = Conversation()

        db_session.add(conversation)
        db_session.flush()

        assert conversation.id is not None
        assert len(conversation.id) == 36
        assert conversation.title == "New Conversation"
        assert conversation.description is None
        assert conversation.created_at is not None
        assert conversation.updated_at is not None

    def test_custom_values(self, db_session):
        conversation = Conversation(
            title="Test Conversation",
            description="A test conversation.",
        )

        db_session.add(conversation)
        db_session.flush()

        assert conversation.title == "Test Conversation"
        assert conversation.description == "A test conversation."


class TestMessageModel:
    def test_message_creation(self, db_session):
        conversation = Conversation(title="Test")
        db_session.add(conversation)
        db_session.flush()

        message = Message(
            conversation_id=conversation.id,
            role="user",
            content="Hello, S7VEN!",
        )

        db_session.add(message)
        db_session.flush()

        assert message.id is not None
        assert message.conversation_id == conversation.id
        assert message.role == "user"
        assert message.content == "Hello, S7VEN!"
        assert message.created_at is not None

    def test_conversation_relationship(self, db_session):
        conversation = Conversation(title="Test")
        message = Message(
            role="assistant",
            content="Hello!",
        )

        conversation.messages.append(message)
        db_session.add(conversation)
        db_session.flush()

        assert message.conversation_id == conversation.id
        assert message.conversation is conversation
        assert message in conversation.messages

    def test_delete_orphan_message(self, db_session):
        conversation = Conversation(title="Test")
        message = Message(
            role="user",
            content="Temporary message",
        )

        conversation.messages.append(message)
        db_session.add(conversation)
        db_session.flush()

        message_id = message.id

        conversation.messages.remove(message)
        db_session.flush()

        assert db_session.get(Message, message_id) is None


class TestMemoryModel:
    def test_default_values(self, db_session):
        memory = Memory(content="User prefers concise answers.")

        db_session.add(memory)
        db_session.flush()

        assert memory.id is not None
        assert len(memory.id) == 36
        assert memory.conversation_id is None
        assert memory.category == "general"
        assert memory.importance == 0.5
        assert memory.created_at is not None
        assert memory.updated_at is not None

    def test_custom_values(self, db_session):
        conversation = Conversation(title="Memory Source")
        db_session.add(conversation)
        db_session.flush()

        memory = Memory(
            conversation_id=conversation.id,
            content="User is building S7VEN.",
            category="project",
            importance=0.9,
        )

        db_session.add(memory)
        db_session.flush()

        assert memory.conversation_id == conversation.id
        assert memory.content == "User is building S7VEN."
        assert memory.category == "project"
        assert memory.importance == 0.9
