import re

import pytest

from loguru import logger


@pytest.mark.parametrize(
    "filter",
    [
        None,
        "",
        "tests",
        "tests.test_add_option_filter",
        (lambda r: True),
        (lambda r: r["level"].name == "DEBUG"),
        {},
        {"": "DEBUG"},
        {"tests": True},
        {"tests.test_add_option_filter": 10},
        {"": "WARNING", "tests": 0},
        {"tests.test_add_option_filter": 5, "tests": False},
        {"tests.test_add_option_filter.foobar": False},
        {"tests.": False},
        {"tests.test_add_option_filter.": False},
    ],
)
def test_filtered_in(filter, writer):
    logger.add(writer, filter=filter, format="{message}")
    logger.debug("Test Filter")
    assert writer.read() == "Test Filter\n"


@pytest.mark.parametrize(
    "filter",
    [
        "test",
        "testsx",
        "tests.",
        "tests.test_add_option_filter.",
        ".",
        (lambda r: False),
        (lambda r: r["level"].no != 10),
        {"": False},
        {"": True, "tests": 50},
        {"tests.test_add_option_filter": False},
        {"tests": "WARNING"},
        {"tests": 5, "tests.test_add_option_filter": 40},
        {"": 100, "tests.test_add_option_filter.foobar": True},
    ],
)
def test_filtered_out(filter, writer):
    logger.add(writer, filter=filter, format="{message}")
    logger.debug("Test Filter")
    assert writer.read() == ""


@pytest.mark.parametrize(
    "filter",
    [
        None,
        lambda _: True,
        {},
        {None: 0},
        {"": False},
        {"tests": False, None: True},
        {"unrelated": 100},
        {None: "INFO", "": "WARNING"},
    ],
)
def test_filtered_in_incomplete_frame_context(writer, filter, incomplete_frame_context):
    logger.add(writer, filter=filter, format="{message}", catch=False)
    logger.info("It's ok")
    assert writer.read() == "It's ok\n"


@pytest.mark.parametrize(
    "filter",
    [
        "tests",
        "",
        lambda _: False,
        {None: False},
        {"": 0, None: "WARNING"},
        {None: 100, "tests": True},
    ],
)
def test_filtered_out_incomplete_frame_context(writer, filter, incomplete_frame_context):
    logger.add(writer, filter=filter, format="{message}", catch=False)
    logger.info("It's not ok")
    assert writer.read() == ""


@pytest.mark.parametrize("filter", [-1, 3.4, object()])
def test_invalid_filter(writer, filter):
    with pytest.raises(TypeError):
        logger.add(writer, filter=filter)


@pytest.mark.parametrize("filter", [{"foo": None}, {"foo": 2.5}, {"a": "DEBUG", "b": object()}])
def test_invalid_filter_dict_level_types(writer, filter):
    with pytest.raises(TypeError):
        logger.add(writer, filter=filter)


@pytest.mark.parametrize("filter", [{1: "DEBUG"}, {object(): 10}])
def test_invalid_filter_dict_module_types(writer, filter):
    with pytest.raises(TypeError):
        logger.add(writer, filter=filter)


@pytest.mark.parametrize("filter", [{"foo": "UNKNOWN_LEVEL"}, {"tests.test_add_option_filter": ""}])
def test_invalid_filter_dict_values_unknown_level(writer, filter):
    with pytest.raises(
        ValueError,
        match=(
            r"The filter dict contains a module '[^']*' associated to "
            r"a level name which does not exist: '[^']*'"
        ),
    ):
        logger.add(writer, filter=filter)


def test_invalid_filter_dict_values_wrong_integer_value(writer):
    with pytest.raises(
        ValueError,
        match=(
            r"The filter dict contains a module '[^']*' associated to an invalid level, "
            r"it should be a positive integer, not: '[^']*'"
        ),
    ):
        logger.add(writer, filter={"tests": -1})


def test_filter_dict_with_custom_level(writer):
    logger.level("MY_LEVEL", 6, color="", icon="")
    logger.add(writer, level=0, filter={"tests": "MY_LEVEL"}, format="{message}")
    logger.log(3, "No")
    logger.log(9, "Yes")
    assert writer.read() == "Yes\n"


def test_invalid_filter_builtin(writer):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The built-in 'filter()' function cannot be used as a 'filter' parameter, this is "
            "most likely a mistake (please double-check the arguments passed to 'logger.add()'"
        ),
    ):
        logger.add(writer, filter=filter)


def broken_filter(record):
    raise RuntimeError("Broken filter")


def test_on_filter_error_reports_failed_handler_and_drops_record(writer):
    received = []
    errors = []

    good_id = logger.add(writer, format="{message}")
    bad_id = logger.add(
        received.append,
        filter=broken_filter,
        on_filter_error=lambda *args: errors.append(args),
        format="{message}",
    )

    logger.info("Hello")

    # The other handler processed the record normally, following its own "catch" policy.
    assert writer.read() == "Hello\n"
    # The record failed for the handler with the broken filter only.
    assert received == []
    # The error was reported exactly once (no recursive re-emission of the exception),
    # with the identity of the failed handler.
    assert len(errors) == 1
    handler_id, record, error = errors[0]
    assert handler_id == bad_id
    assert handler_id != good_id
    assert record["message"] == "Hello"
    assert isinstance(error, RuntimeError)
    assert str(error) == "Broken filter"


def test_on_filter_error_not_affected_by_catch_false(writer):
    errors = []
    logger.add(
        writer,
        filter=broken_filter,
        on_filter_error=lambda *args: errors.append(args),
        catch=False,
        format="{message}",
    )
    logger.info("Hello")
    assert writer.read() == ""
    assert len(errors) == 1


def test_filter_error_default_behavior_caught(writer, capsys):
    logger.add(writer, filter=broken_filter, format="{message}")
    logger.info("Hello")
    assert writer.read() == ""
    out, err = capsys.readouterr()
    assert out == ""
    assert "--- Logging error in Loguru Handler #0 ---" in err
    assert "RuntimeError: Broken filter" in err


def test_filter_error_default_behavior_not_caught(writer):
    logger.add(writer, filter=broken_filter, catch=False, format="{message}")
    with pytest.raises(RuntimeError, match="Broken filter"):
        logger.info("Hello")
    assert writer.read() == ""


@pytest.mark.parametrize("on_filter_error", [-1, 3.4, object(), "foo"])
def test_invalid_on_filter_error(writer, on_filter_error):
    with pytest.raises(TypeError, match="Invalid 'on_filter_error'"):
        logger.add(writer, on_filter_error=on_filter_error)
